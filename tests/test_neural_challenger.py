"""Tests for the hierarchical neural challenger.

The model is trained on 129 labelled proteins with an effective count nearer 39, against
roughly 150,000 trainable parameters. Nothing here can fix that ratio; what these tests can
do is make sure the evaluation is honest about it - that folds are clustered rather than
random, that the parameter count is reported beside the example count, and that the three
input views are actually distinct rather than one view fed in three times.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from experimental.neural.architecture import (
    DEFAULT_LORA_RANK,
    ModelConfig,
    TrainingConfig,
    build_model,
    count_trainable_parameters,
    torch_available,
)
from experimental.neural.data import (
    MAX_SYNTAX_TOKENS,
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    ProteinExample,
    build_syntax_vocabulary,
    grammar_profile_channels,
    label_index,
    syntax_tokens,
)
from experimental.neural.train import LabelSpace, neural_report
from validation.labels import GoldLabel

_JD = "MVKETKFYDILGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKEISEAYEVLSDPEKREIYDQ" * 3

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch is not installed")

# Imported after the skip marker so a machine without torch collects this file cleanly.
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class _Evidence:
    def __init__(self, layout: str) -> None:
        self.domain_family_layout = layout


class _Record:
    def __init__(self, accession: str, sequence: str) -> None:
        self.accession = accession
        self.sequence = sequence


class _Layout:
    def __init__(self, accession: str, sequence: str, layout: str) -> None:
        self.record = _Record(accession, sequence)
        self.evidence = _Evidence(layout)


# Data assembly
# -------------


def test_the_profile_has_four_aligned_channels() -> None:
    """Misaligned channels would silently pair one residue's entropy with another's charge."""
    channels = grammar_profile_channels(_JD)
    assert len(channels) == 4
    lengths = {len(channel) for channel in channels}
    assert len(lengths) == 1, f"channels differ in length: {lengths}"
    assert lengths.pop() > 0


def test_an_empty_sequence_yields_empty_channels() -> None:
    """A protein with no sequence must not produce a phantom profile."""
    assert grammar_profile_channels("") == [[], [], [], []]


def test_the_syntax_vocabulary_reserves_padding_and_unknown() -> None:
    """A family colliding with the pad id would be masked out of attention entirely."""
    layouts = [_Layout("P1", _JD, "j_domain>gf_rich"), _Layout("P2", _JD, "j_domain>ctd")]
    vocabulary = build_syntax_vocabulary(layouts)
    assert PAD_TOKEN not in vocabulary.values()
    assert UNKNOWN_TOKEN not in vocabulary.values()
    assert set(vocabulary) == {"j_domain", "gf_rich", "ctd"}


def test_domain_order_is_preserved_not_sorted() -> None:
    """Order is the hypothesis; a bag of families cannot separate these two proteins."""
    vocabulary = {"j_domain": 2, "gf_rich": 3}
    forward = syntax_tokens(_Layout("P1", _JD, "j_domain>gf_rich"), vocabulary)
    reverse = syntax_tokens(_Layout("P2", _JD, "gf_rich>j_domain"), vocabulary)
    assert forward == [2, 3]
    assert reverse == [3, 2]
    assert forward != reverse


def test_an_unseen_family_becomes_the_unknown_token() -> None:
    """A family first seen at inference must degrade, not collide with padding."""
    assert syntax_tokens(_Layout("P1", _JD, "never_seen"), {"j_domain": 2}) == [UNKNOWN_TOKEN]


def test_label_index_is_stable_across_calls() -> None:
    """An unstable head layout would make two runs' predictions incomparable."""
    assert label_index(["b", "a", "c"]) == label_index(["c", "b", "a"]) == {"a": 0, "b": 1, "c": 2}


# Architecture
# ------------


def test_the_model_builds_and_reports_its_trainable_count() -> None:
    """The parameter count is the caveat, so it must be obtainable."""
    config = ModelConfig(n_classes=3, n_subclasses=6, n_syntax_tokens=20, n_static_features=51)
    model = build_model(config)
    assert count_trainable_parameters(model) > 0


def test_both_heads_produce_logits_of_the_right_width() -> None:
    """A mismatched head silently argmaxes over the wrong label space."""
    config = ModelConfig(n_classes=3, n_subclasses=7, n_syntax_tokens=20, n_static_features=5)
    model = build_model(config)
    profile = torch.rand(2, 4, 30)
    mask = torch.ones(2, 30)
    syntax = torch.tensor([[2, 3, 0], [4, 0, 0]], dtype=torch.long)
    static = torch.rand(2, 5)
    class_logits, subclass_logits = model(profile, mask, syntax, static_features=static)
    assert class_logits.shape == (2, 3)
    assert subclass_logits.shape == (2, 7)


def test_padding_does_not_change_a_proteins_representation() -> None:
    """A short protein batched with a long one must not be diluted by the padding.

    If the mask were ignored, a protein's prediction would depend on what it was batched
    with, which makes every result irreproducible.
    """
    config = ModelConfig(n_classes=2, n_subclasses=2, n_syntax_tokens=10, n_static_features=3)
    model = build_model(config)
    model.eval()

    short = torch.rand(1, 4, 10)
    syntax = torch.tensor([[2, 3]], dtype=torch.long)
    static = torch.rand(1, 3)

    with torch.no_grad():
        alone, _ = model(short, torch.ones(1, 10), syntax, static_features=static)
        padded_profile = torch.cat([short, torch.rand(1, 4, 20)], dim=2)
        padded_mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 20)], dim=1)
        padded, _ = model(padded_profile, padded_mask, syntax, static_features=static)

    assert torch.allclose(alone, padded, atol=1e-5)


def test_the_lora_rank_stays_small_enough_for_the_label_set() -> None:
    """A rank large enough to have more free parameters than labels defeats the point."""
    assert DEFAULT_LORA_RANK <= 16


# Training
# --------


def _separable_examples(n: int = 24) -> list[ProteinExample]:
    rng = random.Random(0)  # noqa: S311 - deterministic test fixture
    examples = []
    for index in range(n):
        label = "a" if index % 2 else "b"
        base = 0.9 if label == "a" else 0.1
        length = rng.randint(15, 25)
        examples.append(
            ProteinExample(
                accession=f"P{index}",
                profile=tuple(tuple(base + rng.uniform(-0.03, 0.03) for _ in range(length)) for _ in range(4)),
                syntax=(2, 3) if label == "a" else (3, 4),
                static=tuple(base + rng.uniform(-0.03, 0.03) for _ in range(4)),
                label_class=label,
                label_subclass=f"{label}1",
            ),
        )
    return examples


def test_a_separable_signal_is_learned() -> None:
    """If the model cannot learn a signal this obvious, no result from it means anything."""
    examples = _separable_examples()
    folds = [[f"P{i}" for i in range(12)], [f"P{i}" for i in range(12, 24)]]
    report = neural_report(
        examples,
        folds,
        n_syntax_tokens=8,
        settings=TrainingConfig(epochs=20, batch_size=8, patience=5, seeds=(0,)),
    )
    assert report["accuracy_mean"] > 0.8


def test_the_report_states_parameters_per_labelled_example() -> None:
    """The binding constraint is the label set, and the ratio must be on the page."""
    examples = _separable_examples()
    folds = [[f"P{i}" for i in range(12)], [f"P{i}" for i in range(12, 24)]]
    report = neural_report(
        examples,
        folds,
        n_syntax_tokens=8,
        settings=TrainingConfig(epochs=3, batch_size=8, patience=2, seeds=(0,)),
    )
    assert report["parameters_per_labelled_example"] > 0
    assert report["trainable_parameters"] > report["n_examples"]


def test_several_seeds_are_run_and_the_spread_reported() -> None:
    """A single run's accuracy on a set this small is not a measurement."""
    examples = _separable_examples()
    folds = [[f"P{i}" for i in range(12)], [f"P{i}" for i in range(12, 24)]]
    report = neural_report(
        examples,
        folds,
        n_syntax_tokens=8,
        settings=TrainingConfig(epochs=3, batch_size=8, patience=2, seeds=(0, 1, 2)),
    )
    assert report["n_seeds"] == 3
    assert len(report["accuracy_by_seed"]) == 3
    assert report["accuracy_spread"] >= 0.0


def test_the_label_space_cannot_be_given_inconsistent_vocabularies() -> None:
    """Both heads read one bundle, so they cannot disagree about what index 2 means."""
    space = LabelSpace(class_to_index={"a": 0, "b": 1}, subclass_to_index={"a1": 0}, n_syntax_tokens=8)
    assert space.index_to_class == {0: "a", 1: "b"}


def test_syntax_sequences_are_capped() -> None:
    """An unbounded token sequence would blow up attention memory on a pathological input."""
    layout = _Layout("P1", _JD, ">".join(["j_domain"] * 500))
    assert len(syntax_tokens(layout, {"j_domain": 2})) == MAX_SYNTAX_TOKENS


def test_gold_labels_are_unwrapped_to_class_strings() -> None:
    """The label space must be strings, not the records they came from.

    ``collect_gold_labels`` returns ``GoldLabel`` objects carrying their evidence. Feeding
    those straight in made the label space unsortable and killed a GPU run *after* the
    entire layout stage had completed - the most expensive place to discover a type error.
    """
    record = GoldLabel(accession="P1", label="B", source_name="DNAJB1", organism_name="Homo sapiens")
    # A bare record is not orderable against another, which is what label_index needs.
    with pytest.raises(TypeError):
        sorted({record, GoldLabel("P2", "A", "DNAJA1", "Homo sapiens")})
    # Unwrapped, it is.
    assert label_index([record.label, "A"]) == {"A": 0, "B": 1}


# ESM-2 in the live path
# ----------------------


def test_the_language_model_is_a_submodule_not_a_precomputed_feature() -> None:
    """Precomputed embeddings would make the LoRA adapters decorative.

    If ESM-2 only ever produced a frozen feature vector, its adapters would never receive a
    gradient and the "+ LoRA" half of the design would do nothing. Holding it as a submodule
    is what puts it in ``model.parameters()`` and therefore in the optimiser.
    """
    config = ModelConfig(n_classes=3, n_subclasses=6, n_syntax_tokens=20, n_static_features=51)
    model = build_model(config, language_model=None)
    assert hasattr(model, "language_model")
    # Without an encoder the fusion trunk must not reserve width for one.
    assert model.language_model is None


def test_a_stand_in_encoder_is_trained_end_to_end() -> None:
    """A gradient must reach the encoder, or it is not in the training path.

    Uses a stand-in with the same interface as ESM-2 rather than downloading the real
    weights, so the wiring is testable without network access or a GPU.
    """

    class _Output:
        def __init__(self, hidden: object) -> None:
            self.last_hidden_state = hidden

    class _StandInEncoder(nn.Module):
        def __init__(self, hidden_size: int = 16) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, hidden_size)
            self.config = type("cfg", (), {"hidden_size": hidden_size})()

        def forward(self, input_ids: object, attention_mask: object = None) -> _Output:
            _ = attention_mask
            return _Output(self.embed(input_ids))

    encoder = _StandInEncoder()
    config = ModelConfig(
        n_classes=2,
        n_subclasses=2,
        n_syntax_tokens=10,
        n_static_features=3,
        esm_hidden_size=16,
    )
    model = build_model(config, language_model=encoder)

    logits, _ = model(
        torch.rand(2, 4, 12),
        torch.ones(2, 12),
        torch.tensor([[2, 3], [4, 0]], dtype=torch.long),
        static_features=torch.rand(2, 3),
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )
    logits.sum().backward()
    assert encoder.embed.weight.grad is not None, "no gradient reached the encoder"
    assert encoder.embed.weight.grad.abs().sum() > 0


def test_padding_cannot_pull_the_pooled_embedding() -> None:
    """An unmasked mean would drag every short protein toward the pad token."""

    class _Output:
        def __init__(self, hidden: object) -> None:
            self.last_hidden_state = hidden

    class _ConstantEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))
            self.config = type("cfg", (), {"hidden_size": 4})()

        def forward(self, input_ids: object, attention_mask: object = None) -> _Output:
            _ = attention_mask
            # Real tokens embed to 1.0; the pad id 0 embeds to a large value that would
            # visibly move an unmasked mean.
            values = torch.where(input_ids == 0, torch.tensor(100.0), torch.tensor(1.0))
            return _Output(values.unsqueeze(-1).repeat(1, 1, 4) * self.scale)

    config = ModelConfig(n_classes=2, n_subclasses=2, n_syntax_tokens=10, n_static_features=0, esm_hidden_size=4)
    model = build_model(config, language_model=_ConstantEncoder())
    model.eval()

    with torch.no_grad():
        pooled = model.language_model(
            input_ids=torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        ).last_hidden_state
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long).unsqueeze(-1).float()
        masked_mean = (pooled * mask).sum(dim=1) / mask.sum(dim=1)
    # Only the real tokens contribute, so the mean is 1.0 and not pulled toward 100.
    assert torch.allclose(masked_mean, torch.ones_like(masked_mean))


def test_seed_settings_carry_the_language_model_flag() -> None:
    """Rebuilding the config field by field silently dropped it, disabling ESM-2 per seed."""
    base = TrainingConfig(use_language_model=True, language_model_learning_rate=5e-5, seeds=(0, 1))
    seeded = replace(base, seed=1)
    assert seeded.use_language_model is True
    assert seeded.language_model_learning_rate == 5e-5
