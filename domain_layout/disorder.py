"""Disorder prediction layer: metapredict when installed, FoldIndex fallback otherwise.

The pipeline needs per-residue disorder to decide which residues are alignable
(structured, MSA route) and which are not (IDR, SHARK route). The real backend is
`metapredict <https://github.com/idptools/metapredict>`_ V3; when it is not installed
(it pulls in PyTorch, which is not always available on a login node) the module falls
back to FoldIndex — a published charge/hydropathy predictor — so the pipeline still
produces a complete, reproducible layout. The backend used is always reported in
``DisorderPrediction.backend`` and in every output table, so results are never silently
attributed to metapredict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from domain_layout.constants import (
    DISORDER_THRESHOLD,
    FOLDINDEX_WINDOW,
    IDR_GAP_CLOSURE,
    MIN_IDR_LENGTH,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

BACKEND_METAPREDICT = "metapredict"
BACKEND_FOLDINDEX = "foldindex"
BACKEND_AUTO = "auto"

# Kyte & Doolittle (1982) hydropathy index.
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

_HYDROPATHY_MIN = -4.5
_HYDROPATHY_RANGE = 9.0
_FOLDINDEX_HYDROPATHY_WEIGHT = 2.785
_FOLDINDEX_OFFSET = 1.151
_POSITIVE_RESIDUES = frozenset("KR")
# metapredict one-hot encodes from exactly these residues and raises on anything else.
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NEGATIVE_RESIDUES = frozenset("DE")


@dataclass(frozen=True, slots=True)
class DisorderPrediction:
    """Per-residue disorder scores and derived IDR / folded intervals.

    Intervals are 1-based and inclusive, matching InterPro domain coordinates.
    """

    sequence_length: int
    scores: tuple[float, ...]
    idr_intervals: tuple[tuple[int, int], ...]
    folded_intervals: tuple[tuple[int, int], ...]
    backend: str

    @property
    def mean_disorder(self) -> float:
        """Mean disorder score across the protein (0.0 for an empty sequence)."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def idr_fraction(self) -> float:
        """Fraction of residues inside a predicted IDR."""
        if self.sequence_length <= 0:
            return 0.0
        covered = sum(end - start + 1 for start, end in self.idr_intervals)
        return covered / self.sequence_length

    def mean_over(self, start: int, end: int) -> float:
        """Mean disorder score over a 1-based inclusive residue interval."""
        window = self.scores[max(0, start - 1) : max(0, end)]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def fraction_disordered_over(self, start: int, end: int, *, threshold: float = DISORDER_THRESHOLD) -> float:
        """Fraction of residues above ``threshold`` in a 1-based inclusive interval."""
        window = self.scores[max(0, start - 1) : max(0, end)]
        if not window:
            return 0.0
        return sum(1 for score in window if score >= threshold) / len(window)

    def residue_is_disordered(self) -> tuple[bool, ...]:
        """Per-residue membership in a predicted IDR interval."""
        flags = [False] * self.sequence_length
        for start, end in self.idr_intervals:
            for position in range(max(1, start), min(self.sequence_length, end) + 1):
                flags[position - 1] = True
        return tuple(flags)


def metapredict_available() -> bool:
    """Whether the metapredict package can be imported."""
    return find_spec("metapredict") is not None


def normalized_hydropathy(residue: str) -> float:
    """Kyte-Doolittle hydropathy rescaled to [0, 1] (unknown residues score 0.5)."""
    value = KYTE_DOOLITTLE.get(residue.upper())
    if value is None:
        return 0.5
    return (value - _HYDROPATHY_MIN) / _HYDROPATHY_RANGE


def residue_charge(residue: str) -> int:
    """Formal charge of a residue at neutral pH (K/R positive, D/E negative)."""
    upper = residue.upper()
    if upper in _POSITIVE_RESIDUES:
        return 1
    if upper in _NEGATIVE_RESIDUES:
        return -1
    return 0


def foldindex_scores(sequence: str, *, window: int = FOLDINDEX_WINDOW) -> tuple[float, ...]:
    """Per-residue disorder scores from FoldIndex (Prilusky et al., 2005).

    FoldIndex computes ``I = 2.785 * <H> - |<R>| - 1.151`` over a sliding window,
    where ``<H>`` is mean normalized hydropathy and ``<R>`` mean net charge.
    ``I <= 0`` indicates disorder; the index is mapped to a ``[0, 1]`` disorder score
    with ``0.5 - I / 2`` so it can be thresholded like a metapredict score.
    """
    if not sequence:
        return ()

    half = max(1, window // 2)
    scores: list[float] = []
    for index in range(len(sequence)):
        start = max(0, index - half)
        end = min(len(sequence), index + half + 1)
        chunk = sequence[start:end]
        mean_hydropathy = sum(normalized_hydropathy(residue) for residue in chunk) / len(chunk)
        mean_charge = sum(residue_charge(residue) for residue in chunk) / len(chunk)
        fold_index = _FOLDINDEX_HYDROPATHY_WEIGHT * mean_hydropathy - abs(mean_charge) - _FOLDINDEX_OFFSET
        scores.append(min(1.0, max(0.0, 0.5 - fold_index / 2.0)))
    return tuple(scores)


def intervals_from_scores(
    scores: Sequence[float],
    *,
    threshold: float = DISORDER_THRESHOLD,
    min_length: int = MIN_IDR_LENGTH,
    gap_closure: int = IDR_GAP_CLOSURE,
) -> tuple[tuple[int, int], ...]:
    """Convert per-residue scores into 1-based inclusive IDR intervals.

    Runs above ``threshold`` are merged across gaps of at most ``gap_closure``
    residues, then runs shorter than ``min_length`` are dropped.
    """
    raw: list[list[int]] = []
    for index, score in enumerate(scores, start=1):
        if score < threshold:
            continue
        if raw and index - raw[-1][1] <= gap_closure + 1:
            raw[-1][1] = index
        else:
            raw.append([index, index])

    return tuple((start, end) for start, end in raw if end - start + 1 >= min_length)


def complement_intervals(intervals: Sequence[tuple[int, int]], length: int) -> tuple[tuple[int, int], ...]:
    """Return the 1-based inclusive intervals not covered by ``intervals``."""
    if length <= 0:
        return ()
    complement: list[tuple[int, int]] = []
    cursor = 1
    for start, end in sorted(intervals):
        if start > cursor:
            complement.append((cursor, min(start - 1, length)))
        cursor = max(cursor, end + 1)
    if cursor <= length:
        complement.append((cursor, length))
    return tuple(item for item in complement if item[0] <= item[1])


def _foldindex_prediction(sequence: str) -> DisorderPrediction:
    scores = foldindex_scores(sequence)
    idr = intervals_from_scores(scores)
    return DisorderPrediction(
        sequence_length=len(sequence),
        scores=scores,
        idr_intervals=idr,
        folded_intervals=complement_intervals(idr, len(sequence)),
        backend=BACKEND_FOLDINDEX,
    )


def _boundaries_to_intervals(boundaries: Any, length: int) -> tuple[tuple[int, int], ...]:  # noqa: ANN401
    """Convert metapredict 0-based half-open boundaries to 1-based inclusive intervals."""
    intervals: list[tuple[int, int]] = []
    for pair in boundaries or []:
        start, end = int(pair[0]), int(pair[1])
        converted = (max(1, start + 1), min(length, end))
        if converted[0] <= converted[1]:
            intervals.append(converted)
    return tuple(sorted(intervals))


def _prediction_from_disorder_object(disorder_object: Any, sequence: str) -> DisorderPrediction:  # noqa: ANN401
    scores = tuple(float(value) for value in disorder_object.disorder)
    length = len(sequence)
    idr = _boundaries_to_intervals(getattr(disorder_object, "disordered_domain_boundaries", ()), length)
    folded = _boundaries_to_intervals(getattr(disorder_object, "folded_domain_boundaries", ()), length)
    return DisorderPrediction(
        sequence_length=length,
        scores=scores,
        idr_intervals=idr,
        folded_intervals=folded or complement_intervals(idr, length),
        backend=BACKEND_METAPREDICT,
    )


def is_metapredict_compatible(sequence: str) -> bool:
    """Whether metapredict can encode every residue of this sequence.

    metapredict one-hot encodes from the 20 standard amino acids and raises on anything
    else. UniProt does contain ``X`` (and occasionally ``U``/``B``/``Z``), so sequences
    are screened before the call rather than after the exception.
    """
    return bool(sequence) and set(sequence) <= _STANDARD_AMINO_ACIDS


def _metapredict_batch(sequences: Mapping[str, str]) -> dict[str, DisorderPrediction] | None:
    """Run one metapredict batch; ``None`` when the call fails outright."""
    try:
        import metapredict  # noqa: PLC0415

        raw = metapredict.predict_disorder(dict(sequences), return_domains=True)
        return {name: _prediction_from_disorder_object(raw[name], sequences[name]) for name in sequences}
    except Exception:  # noqa: BLE001 - third-party backend: degrade instead of aborting the run
        logger.warning("metapredict batch failed; retrying sequence by sequence.", exc_info=True)
        return None


def _metapredict_predictions(sequences: Mapping[str, str]) -> dict[str, DisorderPrediction]:
    """Predict with metapredict where possible, per sequence.

    Two failure modes are contained here rather than allowed to spread:

    * a sequence metapredict cannot encode (non-standard residues) is never sent to it;
    * if a batch still fails, the remaining sequences are retried individually.

    Both matter because predictions are batched: without this, a single unusual sequence
    would silently downgrade every other protein in the same batch to the fallback
    predictor. Sequences that cannot be predicted are simply absent from the result, and
    the caller fills them in with FoldIndex.
    """
    if not metapredict_available():
        return {}

    compatible = {name: sequence for name, sequence in sequences.items() if is_metapredict_compatible(sequence)}
    skipped = len(sequences) - len(compatible)
    if skipped:
        logger.info(
            "%s of %s sequence(s) contain residues metapredict cannot encode; "
            "those use FoldIndex while the rest still use metapredict.",
            skipped,
            len(sequences),
        )
    if not compatible:
        return {}

    batch = _metapredict_batch(compatible)
    if batch is not None:
        return batch

    predictions: dict[str, DisorderPrediction] = {}
    for name, sequence in compatible.items():
        single = _metapredict_batch({name: sequence})
        if single is not None:
            predictions.update(single)
    return predictions


def predict_disorder(sequence: str, *, backend: str = BACKEND_AUTO) -> DisorderPrediction:
    """Predict per-residue disorder for one sequence.

    Args:
        sequence: Amino-acid sequence.
        backend: ``auto`` (metapredict when installed, else FoldIndex), ``metapredict``
            (still degrades to FoldIndex if the call fails), or ``foldindex``.
    """
    if not sequence:
        return DisorderPrediction(
            sequence_length=0,
            scores=(),
            idr_intervals=(),
            folded_intervals=(),
            backend=BACKEND_FOLDINDEX if backend != BACKEND_METAPREDICT else backend,
        )

    return predict_disorder_many({"query": sequence}, backend=backend)["query"]


def predict_disorder_many(
    sequences: Mapping[str, str],
    *,
    backend: str = BACKEND_AUTO,
) -> dict[str, DisorderPrediction]:
    """Predict disorder for many sequences, batching through metapredict when available.

    Sequences are upper-cased first: metapredict encodes residues from an upper-case
    alphabet and rejects or mis-scores lower-case input, and FoldIndex looks hydropathy
    up per residue. Normalizing here keeps both backends agreeing on the same sequence.

    A run may legitimately end up mixing backends — metapredict for most proteins and
    FoldIndex for the few whose sequences it cannot encode. Each prediction records the
    backend that produced it, so the mix is visible per protein rather than averaged away.
    """
    usable = {name: sequence.upper() for name, sequence in sequences.items() if sequence}
    predictions: dict[str, DisorderPrediction] = {
        name: _foldindex_prediction("") for name, sequence in sequences.items() if not sequence
    }

    if usable and backend in {BACKEND_AUTO, BACKEND_METAPREDICT}:
        predictions.update(_metapredict_predictions(usable))

    # Anything metapredict could not handle falls back individually, not in bulk.
    predictions.update(
        {name: _foldindex_prediction(sequence) for name, sequence in usable.items() if name not in predictions},
    )
    return predictions


def active_backend(*, backend: str = BACKEND_AUTO) -> str:
    """Report which disorder backend a run would use."""
    if backend == BACKEND_FOLDINDEX:
        return BACKEND_FOLDINDEX
    return BACKEND_METAPREDICT if metapredict_available() else BACKEND_FOLDINDEX
