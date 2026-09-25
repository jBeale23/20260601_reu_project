"""Assemble the three views each protein presents to the neural challenger.

Every protein becomes:

``profile``
    A four-channel series along the sequence - conditional complexity, windowed Shannon
    entropy, windowed net charge, and windowed hydrophobic fraction. This is the grammar
    view with position retained; the 51-feature vector the classical grammar model uses is
    this series with the positions averaged away.
``syntax``
    The ordered domain-family tokens, ``j_domain>gf_rich>ctd`` becoming ``[3, 7, 5]``. Order
    is the whole point: a bag of families cannot tell ``j_domain>gf_rich`` from
    ``gf_rich>j_domain``, and J-domain position is the incumbent rule system's most-used
    feature.
``static``
    The 51 grammar features, passed through unchanged so the neural model is never handed
    strictly less information than the classical one it is being compared against.

Windowed channels are computed here rather than added to :mod:`domain_layout.grammar`,
which exposes these measures as scalars over a whole sequence. The scalars are the right
interface for the classical model and the wrong one here, and changing them would alter
numbers the existing results depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.grammar import (
    DEFAULT_PROFILE_STEP,
    DEFAULT_PROFILE_WINDOW,
    conditional_complexity_profile,
    nmer_counts,
    shannon_entropy,
)
from experimental.grammar_classifier import features_from_layout
from experimental.neural.architecture import MAX_SEQUENCE_LENGTH

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.pipeline import ProteinLayout

# Positive and negative residues at neutral pH, matching domain_layout.grammar.
_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")
# Kyte-Doolittle positives, collapsed to a set: the hydrophobic channel is a fraction, not
# a mean hydropathy, so a membership test is the honest resolution here.
_HYDROPHOBIC = frozenset("AVILMFWC")

# Longest profile the model sees. Beyond this the sequence is truncated; see
# architecture.MAX_SEQUENCE_LENGTH for the matching cap on the language model.
MAX_PROFILE_WINDOWS = 256

PAD_TOKEN = 0
UNKNOWN_TOKEN = 1
_RESERVED_TOKENS = 2

MAX_SYNTAX_TOKENS = 64


@dataclass(frozen=True, slots=True)
class ProteinExample:
    """One protein, in every view the model consumes."""

    accession: str
    profile: tuple[tuple[float, ...], ...]
    syntax: tuple[int, ...]
    static: tuple[float, ...]
    label_class: str
    label_subclass: str
    # The raw sequence, kept because the language model tokenizes residues directly rather
    # than consuming any of the derived views. Truncated at MAX_SEQUENCE_LENGTH, which
    # trades a little C-terminal signal for a bounded attention footprint.
    sequence: str = ""


def _windowed(sequence: str, window: int, step: int) -> list[str]:
    """Overlapping windows across a sequence, as substrings."""
    if len(sequence) <= window:
        return [sequence] if sequence else []
    return [sequence[start : start + window] for start in range(0, len(sequence) - window + 1, step)]


def grammar_profile_channels(
    sequence: str,
    *,
    window: int = DEFAULT_PROFILE_WINDOW,
    step: int = DEFAULT_PROFILE_STEP,
) -> list[list[float]]:
    """Four aligned channels along the sequence.

    Returns ``[complexity, entropy, net_charge, hydrophobic_fraction]``, each a list of the
    same length. The complexity channel comes from the shared implementation so the neural
    model and the reported complexity profiles cannot drift apart.
    """
    if not sequence:
        return [[], [], [], []]

    windows = _windowed(sequence, window, step)
    entropy = [shannon_entropy(nmer_counts(item, 1)) for item in windows]
    charge = [
        (sum(1 for residue in item if residue in _POSITIVE) - sum(1 for residue in item if residue in _NEGATIVE))
        / max(1, len(item))
        for item in windows
    ]
    hydrophobic = [sum(1 for residue in item if residue in _HYDROPHOBIC) / max(1, len(item)) for item in windows]

    complexity_profile = conditional_complexity_profile(sequence, window=window, step=step)
    complexity = list(complexity_profile.bits_per_residue)

    # The complexity profile starts one window in by construction - its first window would
    # otherwise be charged the compressor's header - so the channels are trimmed to their
    # common length rather than padded, which would invent structure at the N-terminus.
    length = min(len(entropy), len(charge), len(hydrophobic), len(complexity)) if complexity else 0
    if length == 0:
        length = min(len(entropy), len(charge), len(hydrophobic))
        complexity = [0.0] * length
    length = min(length, MAX_PROFILE_WINDOWS)
    return [complexity[:length], entropy[:length], charge[:length], hydrophobic[:length]]


def build_syntax_vocabulary(layouts: Sequence[ProteinLayout]) -> dict[str, int]:
    """Map each domain family to a token id.

    Ids 0 and 1 are reserved for padding and unknown, so a family seen only at inference
    time degrades to a token the model has representation for rather than colliding with
    padding and being masked out.
    """
    families: set[str] = set()
    for layout in layouts:
        layout_string = layout.evidence.domain_family_layout
        families.update(part for part in layout_string.split(">") if part)
    return {family: index + _RESERVED_TOKENS for index, family in enumerate(sorted(families))}


def syntax_tokens(layout: ProteinLayout, vocabulary: Mapping[str, int]) -> list[int]:
    """The domain-family sequence as token ids, in architectural order."""
    parts = [part for part in layout.evidence.domain_family_layout.split(">") if part]
    return [vocabulary.get(part, UNKNOWN_TOKEN) for part in parts][:MAX_SYNTAX_TOKENS]


def build_examples(
    layouts: Sequence[ProteinLayout],
    labels: Mapping[str, str],
    *,
    subclass_labels: Mapping[str, str] | None = None,
    vocabulary: Mapping[str, int] | None = None,
) -> tuple[list[ProteinExample], dict[str, int], list[str]]:
    """Assemble every labelled protein into examples.

    Returns the examples, the syntax vocabulary, and the ordered static feature names, so a
    caller can confirm the neural model saw exactly the features the classical one did.
    """
    vocab = dict(vocabulary) if vocabulary is not None else build_syntax_vocabulary(layouts)
    subclasses = subclass_labels or {}

    feature_names: list[str] = []
    examples: list[ProteinExample] = []
    for layout in layouts:
        accession = layout.record.accession
        if accession not in labels:
            continue
        sequence = layout.record.sequence
        if not sequence:
            continue
        channels = grammar_profile_channels(sequence)
        if not channels[0]:
            continue
        static = features_from_layout(layout)
        if not feature_names:
            feature_names = sorted(static)
        examples.append(
            ProteinExample(
                accession=accession,
                profile=tuple(tuple(channel) for channel in channels),
                syntax=tuple(syntax_tokens(layout, vocab)) or (UNKNOWN_TOKEN,),
                static=tuple(float(static.get(name, 0.0)) for name in feature_names),
                label_class=labels[accession],
                label_subclass=subclasses.get(accession, labels[accession]),
                sequence=sequence[:MAX_SEQUENCE_LENGTH],
            ),
        )
    return examples, vocab, feature_names


def label_index(values: Sequence[str]) -> dict[str, int]:
    """Stable label-to-index mapping, sorted so a rerun reproduces the same head layout."""
    return {label: index for index, label in enumerate(sorted(set(values)))}
