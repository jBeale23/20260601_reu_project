"""Reading a protein as a sentence, and finding where it stops making sense.

:class:`~domain_layout.grammar.CorpusGrammar` scores a whole sequence: one surprisal per
protein, one z-score, one verdict. That answers "is this protein written unusually" and
stops there. It cannot say *where* the unusual writing is, or *what kind* of unusual it is -
and those are the questions that matter if the grammar is to be diagnostic rather than
descriptive.

This module scores position by position against a fitted corpus, finds contiguous runs that
read badly, and classifies each run by the way it is odd.

The error types, and why these
------------------------------
Each corresponds to a distinct thing that can go wrong in a chain, and each has a different
structural consequence:

``low_complexity_insertion``
    A stretch with collapsed alphabet diversity - a run of one or two residue types. In a
    folded domain this breaks packing; in a linker it is unremarkable, which is why the
    surrounding context is reported with the call.
``repeat_expansion``
    A short motif tiled many times. Distinguished from low complexity because the mechanism
    differs: expansions arise from replication slippage and are the basis of an entire
    disease class, and they can be perfectly diverse in composition while still being
    grammatically degenerate.
``charge_anomaly``
    Local net charge far from the family's. Charge drives chaperone-substrate contact, so a
    patch that is strongly acidic or basic where its relatives are not is a candidate
    loss-of-binding. Outright sign reversal is noted in the detail; requiring it as the
    trigger missed every anomaly in a family whose own mean charge is near zero, which is
    most of them.
``hydrophobic_exposure``
    A hydrophobic run where the family is polar. The classic aggregation-prone lesion: a
    sticky patch that should have been buried or absent.
``polar_intrusion``
    The converse - polar residues where the family is hydrophobic, which destabilises a
    core.
``compositional_shift``
    The residue distribution differs from the family's without falling into the above. The
    catch-all, reported as such rather than dressed up.

What this is and is not
-----------------------
This is anomaly detection against a family-conditioned background. Calling a deviation an
"error" is a hypothesis: most will be legitimate variation, since families are diverse and
a corpus is not a rulebook. The word is used because it is the right frame for the
question - and the framing is testable, which is the point. If these spans are meaningful
they should coincide with known pathogenic variants more often than chance, and that test
lives in :mod:`validation.syntax_disease`.

What this module cannot do is say how to fix a span. Suggesting a repair means predicting
the structural consequence of a substitution, which is a folding calculation, not a
grammatical one.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.grammar import DEFAULT_ALPHABET, REDUCED_ALPHABETS, reduce_sequence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.grammar import CorpusGrammar

logger = logging.getLogger(__name__)

# Window over which local surprisal is measured. Short enough to localise a lesion to a
# loop or a strand, long enough that a single substitution does not dominate.
DEFAULT_WINDOW = 15
DEFAULT_STEP = 3

# A window is anomalous above this many standard deviations of the protein's own local
# surprisal. Relative to the protein, not the corpus: an unusual protein should not report
# every one of its windows as an error.
DEFAULT_Z_THRESHOLD = 2.0

# Runs shorter than this are single-window blips rather than spans.
# A lesion shorter than this is a single-residue blip rather than a span the grammar can
# speak to; below roughly this length, composition is not measurable.
MIN_SPAN_RESIDUES = 6

# Fewer windows than this and the median/MAD baseline is not estimable.
MIN_WINDOWS_FOR_BASELINE = 5

# Turns a median absolute deviation into a standard-deviation equivalent, so the threshold
# keeps its usual "sigmas" meaning.
_MAD_TO_SIGMA = 1.4826

# Error type names.
LOW_COMPLEXITY = "low_complexity_insertion"
REPEAT_EXPANSION = "repeat_expansion"
CHARGE_ANOMALY = "charge_anomaly"
HYDROPHOBIC_EXPOSURE = "hydrophobic_exposure"
POLAR_INTRUSION = "polar_intrusion"
COMPOSITIONAL_SHIFT = "compositional_shift"

ERROR_TYPES = (
    LOW_COMPLEXITY,
    REPEAT_EXPANSION,
    CHARGE_ANOMALY,
    HYDROPHOBIC_EXPOSURE,
    POLAR_INTRUSION,
    COMPOSITIONAL_SHIFT,
)

_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")
_HYDROPHOBIC = frozenset("AVILMFWC")

# Distinct residue types below this share of the window means collapsed diversity.
_LOW_COMPLEXITY_DIVERSITY = 0.35
# A motif tiling this much of a window makes it a repeat rather than merely low-complexity.
_REPEAT_COVERAGE = 0.6
_REPEAT_UNIT_MAX = 5
# Charge or hydrophobicity must differ from the family by this much to be called inverted.
_CHARGE_DELTA = 0.25
_HYDROPHOBIC_DELTA = 0.25


@dataclass(frozen=True, slots=True)
class GrammaticalLesion:
    """One span that reads badly, and how.

    Called a lesion rather than an error because ``SyntaxError`` is a built-in, and a class
    shadowing it would be a genuinely confusing collision in a traceback.
    """

    start: int
    end: int
    error_type: str
    z_score: float
    surprisal_per_residue: float
    subsequence: str
    detail: str

    @property
    def length(self) -> int:
        """Residues covered, 1-based inclusive."""
        return self.end - self.start + 1

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "error_type": self.error_type,
            "z_score": round(self.z_score, 3),
            "surprisal_per_residue": round(self.surprisal_per_residue, 4),
            "subsequence": self.subsequence,
            "detail": self.detail,
        }


def positional_surprisal(
    sequence: str,
    corpus: CorpusGrammar,
    *,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> list[tuple[int, float]]:
    """Surprisal per residue in each window, as ``(start, bits)`` pairs.

    Uses the corpus's own n-mer model, so a window is scored against how the family writes
    rather than against a uniform expectation.
    """
    if not corpus.is_fitted or len(sequence) < window:
        return []
    reduced = reduce_sequence(sequence, alphabet=getattr(corpus, "alphabet", DEFAULT_ALPHABET))
    scores: list[tuple[int, float]] = []
    for start in range(0, len(reduced) - window + 1, step):
        piece = reduced[start : start + window]
        scores.append((start, corpus._surprisal(piece) / window))  # noqa: SLF001 - same package
    return scores


def _diversity(piece: str) -> float:
    """Distinct residue types over length."""
    return len(set(piece)) / len(piece) if piece else 0.0


def _repeat_unit(piece: str) -> tuple[str, float]:
    """The short motif tiling a span, and how much of the span it covers."""
    best, coverage = "", 0.0
    for size in range(1, min(_REPEAT_UNIT_MAX, len(piece) // 2) + 1):
        unit = piece[:size]
        tiled = (unit * (len(piece) // size + 1))[: len(piece)]
        matched = sum(1 for a, b in zip(piece, tiled, strict=True) if a == b) / len(piece)
        if matched > coverage:
            best, coverage = unit, matched
    return best, coverage


def _fraction(piece: str, residues: frozenset[str]) -> float:
    """Share of a span made of the given residues."""
    return sum(1 for residue in piece if residue in residues) / len(piece) if piece else 0.0


def _net_charge(piece: str) -> float:
    """Net charge per residue."""
    return _fraction(piece, _POSITIVE) - _fraction(piece, _NEGATIVE)


def classify_error(
    span: str,
    *,
    family_charge: float = 0.0,
    family_hydrophobic: float = 0.0,
) -> tuple[str, str]:
    """What kind of error a span is, and a one-line account of why.

    Ordered by consequence, not by how obvious the pattern is. A poly-aspartate tract is
    simultaneously a repeat, a low-complexity stretch, and a charge inversion; the call
    worth making is the one that says what it does to the protein.
    """
    unit, coverage = _repeat_unit(span)
    if coverage >= _REPEAT_COVERAGE and len(unit) <= _REPEAT_UNIT_MAX:
        return REPEAT_EXPANSION, f"motif {unit!r} tiles {coverage:.0%} of the span"

    # Charge and hydrophobicity come before low complexity, because a poly-aspartate tract
    # is both - and the functional consequence is the charge, not the collapsed alphabet.
    # Calling it "low complexity" would describe the span accurately and say nothing about
    # why it matters.
    charge = _net_charge(span)
    # Magnitude of difference, not a sign flip. Requiring opposite signs meant a strongly
    # acidic patch inside a near-neutral family never triggered - both are negative, so the
    # product test passed and the span fell through to "low complexity", which describes it
    # without saying what it does. Sign reversal is reported in the detail when it happens.
    if abs(charge - family_charge) >= _CHARGE_DELTA:
        direction = "reversed" if charge * family_charge < 0 else "shifted"
        return CHARGE_ANOMALY, f"net charge {charge:+.2f} {direction} against family {family_charge:+.2f}"

    hydrophobic = _fraction(span, _HYDROPHOBIC)
    if hydrophobic - family_hydrophobic >= _HYDROPHOBIC_DELTA:
        return HYDROPHOBIC_EXPOSURE, f"hydrophobic fraction {hydrophobic:.2f} against family {family_hydrophobic:.2f}"
    if family_hydrophobic - hydrophobic >= _HYDROPHOBIC_DELTA:
        return POLAR_INTRUSION, f"hydrophobic fraction {hydrophobic:.2f} against family {family_hydrophobic:.2f}"

    diversity = _diversity(span)
    if diversity <= _LOW_COMPLEXITY_DIVERSITY:
        common = max(set(span), key=span.count)
        return LOW_COMPLEXITY, f"{len(set(span))} residue types over {len(span)}; {common!r} dominates"

    return COMPOSITIONAL_SHIFT, f"diversity {diversity:.2f}, net charge {charge:+.2f}, no specific lesion"


def family_baseline(sequences: Sequence[str]) -> tuple[float, float]:
    """Mean net charge and hydrophobic fraction across a family.

    The reference an error is called against. Without it, "charge inversion" would mean
    "differs from neutral", which is a statement about the alphabet rather than the family.
    """
    if not sequences:
        return 0.0, 0.0
    charges = [_net_charge(item) for item in sequences if item]
    hydrophobics = [_fraction(item, _HYDROPHOBIC) for item in sequences if item]
    return (
        statistics.fmean(charges) if charges else 0.0,
        statistics.fmean(hydrophobics) if hydrophobics else 0.0,
    )


@dataclass(frozen=True, slots=True)
class FamilyReference:
    """What the family normally looks like, which is what a lesion is called against.

    Without it, "charge inversion" would mean "differs from neutral" - a statement about
    the alphabet rather than about this protein's relatives.
    """

    net_charge: float = 0.0
    hydrophobic_fraction: float = 0.0

    @classmethod
    def from_sequences(cls, sequences: Sequence[str]) -> FamilyReference:
        """Fit the reference from a family's sequences."""
        charge, hydrophobic = family_baseline(sequences)
        return cls(net_charge=charge, hydrophobic_fraction=hydrophobic)


DEFAULT_FAMILY = FamilyReference()


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    """Window geometry and the threshold a span must clear."""

    window: int = DEFAULT_WINDOW
    step: int = DEFAULT_STEP
    z_threshold: float = DEFAULT_Z_THRESHOLD


DEFAULT_SETTINGS = DetectorSettings()


def _residue_scores(
    sequence: str,
    scores: Sequence[tuple[int, float]],
    *,
    centre: float,
    spread: float,
    window: int,
) -> list[float]:
    """Spread each window's z-score over the residues it covers.

    Per-residue rather than per-window. Taking the union of flagged windows reported a
    twenty-residue insertion as a thirty-three-residue span, because a window flags whenever
    the lesion falls anywhere inside it - so the span bled a full window past each edge, and
    the classifier then saw the lesion diluted by flanking native sequence.
    """
    totals = [0.0] * len(sequence)
    coverage = [0] * len(sequence)
    for start_index, value in scores:
        z = (value - centre) / spread
        for position in range(start_index, min(len(sequence), start_index + window)):
            totals[position] += z
            coverage[position] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, coverage, strict=True)]


def _contiguous_spans(averaged: Sequence[float], threshold: float) -> list[tuple[int, int]]:
    """Runs where the per-residue evidence stays above the threshold."""
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(averaged):
        if averaged[position] >= threshold:
            begin = position
            while position < len(averaged) and averaged[position] >= threshold:
                position += 1
            spans.append((begin, position))
        else:
            position += 1
    return spans


def find_syntax_errors(
    sequence: str,
    corpus: CorpusGrammar,
    *,
    family: FamilyReference = DEFAULT_FAMILY,
    settings: DetectorSettings = DEFAULT_SETTINGS,
) -> list[GrammaticalLesion]:
    """Locate and classify the spans of a sequence that read badly.

    Windows are scored against the corpus, then standardised against *this protein's own*
    distribution of window scores. That choice matters: standardising against the corpus
    would make every window of an unusual protein an error, which localises nothing.
    """
    window, step, z_threshold = settings.window, settings.step, settings.z_threshold
    scores = positional_surprisal(sequence, corpus, window=window, step=step)
    # A handful of windows cannot establish a baseline to be unusual against.
    if len(scores) < MIN_WINDOWS_FOR_BASELINE:
        return []

    values = [value for _start, value in scores]
    # Median and MAD, not mean and standard deviation. A long lesion occupies many
    # windows and drags a mean baseline up toward itself, so the lesion stops looking
    # unusual relative to its own protein - a twenty-residue polyglutamine tract, the
    # canonical repeat-expansion mechanism, went entirely undetected that way. The
    # median is unmoved by a minority of extreme windows, which is the contamination here.
    centre = statistics.median(values)
    spread = statistics.median([abs(value - centre) for value in values]) * _MAD_TO_SIGMA
    if spread <= 0:
        return []

    # Per-residue attribution, not per-window. Merging flagged windows and taking their
    # union reported a twenty-residue insertion as a thirty-three-residue span, because a
    # window flags whenever the lesion falls anywhere inside it - so the span bled a full
    # window past each edge and the classifier then saw the lesion diluted by flanking
    # native sequence. Spreading each window's score over the residues it covers, and
    # cutting where the per-residue evidence falls back, localises the lesion itself.
    averaged = _residue_scores(sequence, scores, centre=centre, spread=spread, window=window)
    spans = _contiguous_spans(averaged, z_threshold)

    errors: list[GrammaticalLesion] = []
    for begin, finish in spans:
        if finish - begin < MIN_SPAN_RESIDUES:
            continue
        piece = sequence[begin:finish]
        if not piece:
            continue
        error_type, detail = classify_error(
            piece,
            family_charge=family.net_charge,
            family_hydrophobic=family.hydrophobic_fraction,
        )
        errors.append(
            GrammaticalLesion(
                start=begin + 1,
                end=finish,
                error_type=error_type,
                z_score=max(averaged[begin:finish]),
                surprisal_per_residue=statistics.fmean(
                    [value for pos, value in enumerate(averaged) if begin <= pos < finish],
                ),
                subsequence=piece[:60],
                detail=detail,
            ),
        )
    return errors


def error_summary(errors: Mapping[str, Sequence[GrammaticalLesion]]) -> dict[str, object]:
    """Aggregate typed errors across many proteins."""
    by_type: dict[str, int] = dict.fromkeys(ERROR_TYPES, 0)
    lengths: list[int] = []
    for found in errors.values():
        for item in found:
            by_type[item.error_type] = by_type.get(item.error_type, 0) + 1
            lengths.append(item.length)
    with_errors = sum(1 for found in errors.values() if found)
    return {
        "n_proteins": len(errors),
        "n_proteins_with_errors": with_errors,
        "n_errors": sum(by_type.values()),
        "errors_by_type": by_type,
        "median_error_length": statistics.median(lengths) if lengths else 0,
        "alphabets_available": sorted(REDUCED_ALPHABETS),
        "note": (
            "Spans are standardised against each protein's own window distribution, so a "
            "globally unusual protein does not report every window as an error. Calling a "
            "span an error is a hypothesis, not a diagnosis: most will be legitimate family "
            "variation. The claim is testable - if these spans mean anything, they should "
            "coincide with known pathogenic variants more often than chance."
        ),
    }
