"""The hierarchical neural challenger.

Three encoders over three views of the same protein, fused into two heads.

``ESM-2 + LoRA``
    A protein language model over the raw sequence. The base weights stay frozen and only
    low-rank adapters train, which is what makes a 650M-parameter encoder trainable against
    a label set of this size at all - full fine-tuning would have orders of magnitude more
    free parameters than labelled examples.
``CNN + BiLSTM over the grammar profile``
    The n-mer syntax view: entropy, order score, and charge profiles computed along the
    sequence. A convolution picks up local motifs in that profile and the recurrent layer
    carries them along the chain, which is the part a bag-of-features grammar model throws
    away by averaging.
``Syntax transformer over the domain-token sequence``
    The architecture itself - the ordered list of domain families, ``j_domain>gf_rich>ctd``
    and so on - treated as a short sentence. Self-attention over it is the piece that can
    represent *order* and *adjacency* of domains rather than only their presence, which is
    the hypothesis the whole project rests on.

Dual heads predict class and subclass from the fused representation, sharing a trunk so the
coarse label regularises the fine one.

A caveat that belongs in the source, not only in a write-up
-----------------------------------------------------------
This is trained on 129 labelled proteins whose effective count after collapsing ortholog
groups is closer to 39. That is far below what a model of this capacity needs, and it is
the reason the incumbent rule system and a profile HMM remain hard to beat here. The
architecture is not the bottleneck; the label set is. Every result from this model should
be read against :mod:`validation.capacity`, which quantifies exactly how fine a distinction
129 labels can support.

Torch is imported lazily so the rest of the package - and the test suite - runs on a machine
without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# The smallest ESM-2 that still carries useful structure signal. Deliberately not the 650M
# model: with 129 labels the larger encoder buys nothing measurable and costs an order of
# magnitude in memory and time. Override for a run that wants to test that claim.
DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"

# LoRA rank. Low, because the adapter must not have more free parameters than the label set
# can constrain; r=8 over the attention projections is a few hundred thousand at most.
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1

# Sequence cap. J-domain proteins run long in the tail; truncating past this trades a little
# C-terminal signal for a bounded memory footprint.
MAX_SEQUENCE_LENGTH = 1024

# Grammar profile channels: entropy, order score, net charge, charged fraction.
GRAMMAR_CHANNELS = 4
GRAMMAR_CONV_WIDTH = 9
GRAMMAR_HIDDEN = 64

SYNTAX_EMBED = 64
SYNTAX_HEADS = 4
SYNTAX_LAYERS = 2
SYNTAX_FEEDFORWARD = 128

FUSION_HIDDEN = 256
DROPOUT = 0.3


class TorchUnavailableError(RuntimeError):
    """Raised when the neural challenger is asked to build without torch installed."""


def torch_available() -> bool:
    """Whether torch can be imported in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def transformers_available() -> bool:
    """Whether the protein language model stack is installed."""
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape and capacity of one neural challenger."""

    n_classes: int
    n_subclasses: int
    n_syntax_tokens: int
    esm_model: str = DEFAULT_ESM_MODEL
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    grammar_channels: int = GRAMMAR_CHANNELS
    grammar_hidden: int = GRAMMAR_HIDDEN
    syntax_embed: int = SYNTAX_EMBED
    syntax_heads: int = SYNTAX_HEADS
    syntax_layers: int = SYNTAX_LAYERS
    fusion_hidden: int = FUSION_HIDDEN
    dropout: float = DROPOUT
    use_language_model: bool = True
    # Static feature vector width (the 51 grammar features), fused alongside the encoders.
    n_static_features: int = 0
    # Hidden width of the language model, which the fusion trunk must account for. 480 for
    # esm2_t12_35M; read off the loaded model rather than assumed when one is supplied.
    esm_hidden_size: int = 480


@dataclass
class TrainingConfig:
    """Optimisation settings for one run."""

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 8
    seed: int = 0
    device: str = "auto"
    # Class weights counter the imbalance in the labelled set, where the largest class is
    # several times the smallest; without them the model learns the prior and stops.
    balance_classes: bool = True
    seeds: Sequence[int] = field(default_factory=lambda: (0, 1, 2))
    # Whether ESM-2 participates in training. Off degrades to the grammar and syntax
    # encoders alone, which is the ablation measuring what the language model contributes.
    use_language_model: bool = True
    # A far smaller learning rate for the adapters than for the heads: the pretrained
    # representation is the asset, and the label set is too small to justify moving it far.
    language_model_learning_rate: float = 5e-5


def build_model(config: ModelConfig, *, language_model: Any = None) -> Any:
    """Construct the network.

    Imported lazily and constructed here rather than at module scope so that importing this
    module - which the test suite does - never requires torch.

    ``language_model`` is the LoRA-adapted ESM-2 from :func:`build_language_model`. It is
    held as a submodule rather than used to precompute embeddings, because precomputed
    embeddings are frozen by construction: the adapters would never receive a gradient and
    the "+ LoRA" half of the design would be decorative. Passing ``None`` trains the grammar
    and syntax encoders alone, which is the ablation that says what the language model adds.
    """
    if not torch_available():
        message = "torch is not installed; the neural challenger cannot be built"
        raise TorchUnavailableError(message)

    import torch
    from torch import nn

    class GrammarEncoder(nn.Module):
        """Convolution over the grammar profile, then a bidirectional recurrence.

        The convolution is what finds a local pattern in the profile - a run of low-entropy,
        high-charge residues, say - and the recurrence is what lets its position relative to
        the rest of the chain matter.
        """

        def __init__(self, channels: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=GRAMMAR_CONV_WIDTH, padding=GRAMMAR_CONV_WIDTH // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.recurrent = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
            self.output_dim = hidden

        def forward(self, profile: Any, mask: Any) -> Any:
            """Encode a batch of grammar profiles of shape (batch, channels, length).

            The recurrence is run over packed sequences rather than the padded tensor. A
            masked mean at the end is not sufficient on its own: the backward half of a
            bidirectional LSTM begins at the last timestep, so on a padded batch it starts
            inside the padding and carries it into the hidden states at real positions.
            The representation would then depend on what a protein happened to be batched
            with, which makes every prediction irreproducible. Packing makes the recurrence
            stop at each sequence's true end.
            """
            features = self.conv(profile * mask.unsqueeze(1)).transpose(1, 2)
            lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.recurrent(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=features.size(1),
            )
            # Masked mean over real positions only, so padding cannot dilute a short protein.
            weights = mask.unsqueeze(-1).float()
            return (output * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    class SyntaxEncoder(nn.Module):
        """Self-attention over the domain-token sequence.

        This is the piece that can represent domain *order*. A bag of domain families cannot
        distinguish ``j_domain>gf_rich`` from ``gf_rich>j_domain``, and the position of the
        J-domain is the single most-used feature in the incumbent rule system.
        """

        def __init__(self, n_tokens: int, embed: int, heads: int, layers: int, dropout: float) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_tokens, embed, padding_idx=0)
            self.position = nn.Parameter(torch.zeros(1, 64, embed))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed,
                nhead=heads,
                dim_feedforward=SYNTAX_FEEDFORWARD,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output_dim = embed

        def forward(self, tokens: Any) -> Any:
            """Encode a batch of padded token id sequences of shape (batch, length)."""
            padding_mask = tokens == 0
            embedded = self.embedding(tokens)
            embedded = embedded + self.position[:, : embedded.size(1), :]
            encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
            keep = (~padding_mask).unsqueeze(-1).float()
            return (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    class HierarchicalJDPModel(nn.Module):
        """The full challenger: three encoders, a fused trunk, and two heads."""

        def __init__(self, model_config: ModelConfig, encoder: Any = None) -> None:
            super().__init__()
            self.config = model_config
            # Registered as a submodule so its trainable LoRA parameters are picked up by
            # the optimiser and moved with .to(device) alongside everything else.
            self.language_model = encoder
            self.grammar = GrammarEncoder(
                model_config.grammar_channels,
                model_config.grammar_hidden,
                model_config.dropout,
            )
            self.syntax = SyntaxEncoder(
                model_config.n_syntax_tokens,
                model_config.syntax_embed,
                model_config.syntax_heads,
                model_config.syntax_layers,
                model_config.dropout,
            )
            fused = self.grammar.output_dim + self.syntax.output_dim
            fused += model_config.n_static_features
            if encoder is not None:
                fused += model_config.esm_hidden_size
                # Mean-pooled residue embeddings arrive on a different scale from the
                # standardised grammar features; normalising keeps the trunk from being
                # dominated by whichever happens to have the larger magnitude.
                self.esm_norm = nn.LayerNorm(model_config.esm_hidden_size)
            self.trunk = nn.Sequential(
                nn.LayerNorm(fused),
                nn.Linear(fused, model_config.fusion_hidden),
                nn.GELU(),
                nn.Dropout(model_config.dropout),
            )
            # Two heads on a shared trunk: the coarse label regularises the fine one, and a
            # subclass prediction inconsistent with its class is penalised through the trunk.
            self.class_head = nn.Linear(model_config.fusion_hidden, model_config.n_classes)
            self.subclass_head = nn.Linear(model_config.fusion_hidden, model_config.n_subclasses)

        def forward(
            self,
            profile: Any,
            profile_mask: Any,
            syntax_tokens: Any,
            *,
            static_features: Any | None = None,
            input_ids: Any | None = None,
            attention_mask: Any | None = None,
        ) -> tuple[Any, Any]:
            """Return class and subclass logits for a batch."""
            parts = [self.grammar(profile, profile_mask), self.syntax(syntax_tokens)]
            if static_features is not None and self.config.n_static_features:
                parts.append(static_features)
            if self.language_model is not None and input_ids is not None:
                hidden_states = self.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                # Masked mean over real tokens. An unmasked mean would let padding pull
                # every short protein's embedding toward the pad token's representation.
                weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
                parts.append(self.esm_norm(pooled))
            hidden = self.trunk(torch.cat(parts, dim=-1))
            return self.class_head(hidden), self.subclass_head(hidden)

    return HierarchicalJDPModel(config, language_model)


def build_language_model(config: ModelConfig) -> Any:
    """Load ESM-2 and wrap its attention projections in LoRA adapters.

    Separated from :func:`build_model` because it is the only part needing network access
    and a GPU to be worth running; a run without it still trains the grammar and syntax
    encoders, which is the ablation that says how much the language model actually adds.
    """
    if not transformers_available():
        message = "transformers and peft are required for the language-model encoder"
        raise TorchUnavailableError(message)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.esm_model)
    base = AutoModel.from_pretrained(config.esm_model)
    for parameter in base.parameters():
        parameter.requires_grad = False
    adapter = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        # ESM-2's attention projection names. Restricting adaptation to attention rather
        # than every linear layer keeps the trainable count in the low hundreds of
        # thousands, which is the most this label set can justify.
        target_modules=["query", "key", "value"],
    )
    return tokenizer, get_peft_model(base, adapter)


def count_trainable_parameters(model: Any) -> int:
    """Trainable parameter count, for reporting against the label count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
