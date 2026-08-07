"""Alignment-free similarity layer: SHARK when installed, BLOSUM k-mer fallback otherwise.

Unalignable regions (IDRs, G/F-rich low-complexity segments, linkers) cannot be scored
by an MSA, so they are compared with `SHARK <https://git.mpi-cbg.de/tothpetroczylab/shark>`_
(``pip install bio-shark``), which scores similarity by relating k-mers.

When ``bio_shark`` is not installed, this module computes an equivalent-in-spirit
SHARK-score (T=x): for every query k-mer, the best BLOSUM62-normalized similarity to any
target k-mer is kept when it clears the threshold, and the score is the symmetrized mean.
The active backend is reported in every output table, so a fallback score is never
presented as a SHARK-score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from domain_layout.constants import DEFAULT_SHARK_K, DEFAULT_SHARK_THRESHOLD, DIVE_K_VALUES

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

BACKEND_BIO_SHARK = "bio_shark"
BACKEND_KMER = "blosum_kmer"
BACKEND_AUTO = "auto"

_MIN_SIMILARITY = 0.0
_MAX_SIMILARITY = 1.0

# bio_shark's substitution matrix covers only the twenty standard residues. Handed a
# sequence containing X (unknown), U (selenocysteine), or the ambiguity codes B and Z, it
# writes one error line per offending k-mer *pair* to stdout and drops those pairs from
# the score. A single such region among a few hundred thousand proteins produced a 191 MB
# log on a full run, and — worse — returned a quietly wrong score. Sequences that fail
# this check are routed to the BLOSUM k-mer backend, whose matrix does contain X, B and Z.
_SHARK_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Cache of encoded k-mers; reference regions are re-scored against every protein.
_KMER_ENCODING_CACHE_SIZE = 4096
# Cap on similarity-matrix cells computed at once, so a very long IDR pair cannot
# allocate an unbounded block.
_MAX_SIMILARITY_BLOCK = 4_000_000


@dataclass(frozen=True, slots=True)
class SharkMatch:
    """Best alignment-free match of one region against a set of reference regions."""

    reference_id: str
    score: float
    backend: str


def shark_available() -> bool:
    """Whether the ``bio_shark`` package can be imported."""
    return find_spec("bio_shark") is not None


def is_shark_compatible(sequence: str) -> bool:
    """Whether ``bio_shark`` can score this sequence without dropping k-mers.

    Assumes the sequence is already upper-cased, as every caller in this module does
    before scoring.
    """
    return bool(sequence) and set(sequence) <= _SHARK_ALPHABET


def active_backend(*, backend: str = BACKEND_AUTO) -> str:
    """Report which similarity backend a run would use."""
    if backend == BACKEND_KMER:
        return BACKEND_KMER
    return BACKEND_BIO_SHARK if shark_available() else BACKEND_KMER


def resolved_backend_for(first: str, second: str, *, backend: str = BACKEND_AUTO) -> str:
    """Which backend will actually score this specific pair.

    Differs from :func:`active_backend` when either sequence carries a non-standard
    residue, since that pair falls back to the k-mer backend regardless of what is
    installed. Reporting the run-wide backend for such a pair would be inaccurate.
    """
    if active_backend(backend=backend) == BACKEND_KMER:
        return BACKEND_KMER
    both_scorable = is_shark_compatible(first.upper()) and is_shark_compatible(second.upper())
    return BACKEND_BIO_SHARK if both_scorable else BACKEND_KMER


@lru_cache(maxsize=1)
def _blosum62_matrix() -> tuple[np.ndarray, dict[str, int]]:
    """Load BLOSUM62 as a dense array plus a residue-to-row index.

    An array lets a whole k-mer block be scored with vectorized gathers instead of one
    dictionary lookup per residue pair, which is the difference between hours and
    minutes on a proteome-scale run.
    """
    from Bio.Align import substitution_matrices  # noqa: PLC0415

    matrix = substitution_matrices.load("BLOSUM62")
    alphabet = [str(letter) for letter in matrix.alphabet]
    index = {letter: position for position, letter in enumerate(alphabet)}
    size = len(alphabet) + 1  # trailing row/column for residues outside the alphabet
    dense = np.zeros((size, size), dtype=np.float64)
    for row, first in enumerate(alphabet):
        for column, second in enumerate(alphabet):
            dense[row, column] = float(matrix[first, second])
    return dense, index


def _substitution_score(first: str, second: str) -> float:
    """BLOSUM62 score for one residue pair; residues outside the alphabet score 0."""
    dense, index = _blosum62_matrix()
    unknown = dense.shape[0] - 1
    return float(dense[index.get(first, unknown), index.get(second, unknown)])


def kmer_similarity(first: str, second: str) -> float:
    """Normalized BLOSUM62 similarity between two equal-length k-mers, in [0, 1].

    Uses the self-similarity-normalized (cosine-like) form
    ``sum(s(a_i, b_i)) / sqrt(sum(s(a_i, a_i)) * sum(s(b_i, b_i)))`` and clips negatives
    to zero, so identical k-mers score 1.0 and dissimilar ones score near 0.
    """
    if not first or len(first) != len(second):
        return _MIN_SIMILARITY
    if first == second:
        return _MAX_SIMILARITY

    cross = sum(_substitution_score(a, b) for a, b in zip(first, second, strict=True))
    self_first = sum(_substitution_score(a, a) for a in first)
    self_second = sum(_substitution_score(b, b) for b in second)
    if self_first <= 0 or self_second <= 0:
        return _MIN_SIMILARITY

    normalized = cross / ((self_first * self_second) ** 0.5)
    return min(_MAX_SIMILARITY, max(_MIN_SIMILARITY, normalized))


def _unique_kmers(sequence: str, k: int) -> tuple[str, ...]:
    if k <= 0 or len(sequence) < k:
        return ()
    seen: dict[str, None] = {}
    for index in range(len(sequence) - k + 1):
        seen.setdefault(sequence[index : index + k], None)
    return tuple(seen)


@lru_cache(maxsize=_KMER_ENCODING_CACHE_SIZE)
def _encode_kmers(sequence: str, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Encode a sequence's unique k-mers as residue indices plus their self-scores.

    Cached because reference regions are re-scored against every query protein; without
    the cache each reference would be re-encoded hundreds of thousands of times.
    """
    kmers = _unique_kmers(sequence, k)
    if not kmers:
        return None

    dense, index = _blosum62_matrix()
    unknown = dense.shape[0] - 1
    encoded = np.array(
        [[index.get(residue, unknown) for residue in kmer] for kmer in kmers],
        dtype=np.intp,
    )
    self_scores = dense[encoded, encoded].sum(axis=1)
    return encoded, self_scores


def _similarity_block(
    query: np.ndarray,
    query_self: np.ndarray,
    target: np.ndarray,
    target_self: np.ndarray,
) -> np.ndarray:
    """Normalized similarity for every query/target k-mer pair, as a 2-D array."""
    dense, _index = _blosum62_matrix()
    cross = np.zeros((query.shape[0], target.shape[0]), dtype=np.float64)
    for position in range(query.shape[1]):
        cross += dense[np.ix_(query[:, position], target[:, position])]

    norm = np.sqrt(np.outer(query_self, target_self))
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(norm > 0, cross / norm, 0.0)
    return np.clip(similarity, _MIN_SIMILARITY, _MAX_SIMILARITY)


def _directional_means(similarity: np.ndarray, threshold: float) -> tuple[float, float]:
    """Mean best-match score in each direction, counting only matches above threshold."""
    forward_best = similarity.max(axis=1)
    backward_best = similarity.max(axis=0)
    forward = float(np.where(forward_best >= threshold, forward_best, 0.0).mean())
    backward = float(np.where(backward_best >= threshold, backward_best, 0.0).mean())
    return forward, backward


def kmer_shark_score(
    first: str,
    second: str,
    *,
    k: int = DEFAULT_SHARK_K,
    threshold: float = DEFAULT_SHARK_THRESHOLD,
) -> float:
    """Symmetrized SHARK-score (T=x) style similarity computed without ``bio_shark``.

    For every k-mer of one sequence the best-scoring k-mer of the other is kept when it
    clears ``threshold``; the two directions are averaged so the score is symmetric.
    """
    query = _encode_kmers(first.upper(), k)
    target = _encode_kmers(second.upper(), k)
    if query is None or target is None:
        return _MIN_SIMILARITY

    query_encoded, query_self = query
    target_encoded, target_self = target

    forward_total = 0.0
    backward_best_overall: np.ndarray | None = None
    rows = max(1, _MAX_SIMILARITY_BLOCK // max(1, target_encoded.shape[0]))
    for start in range(0, query_encoded.shape[0], rows):
        stop = start + rows
        block = _similarity_block(
            query_encoded[start:stop],
            query_self[start:stop],
            target_encoded,
            target_self,
        )
        block_forward, _ = _directional_means(block, threshold)
        forward_total += block_forward * block.shape[0]
        column_best = block.max(axis=0)
        backward_best_overall = (
            column_best if backward_best_overall is None else np.maximum(backward_best_overall, column_best)
        )

    forward = forward_total / query_encoded.shape[0]
    backward = _thresholded_mean(backward_best_overall, threshold)
    return (forward + backward) / 2.0


def _thresholded_mean(values: np.ndarray, threshold: float) -> float:
    """Mean of best-match scores, counting only those at or above the threshold."""
    return float(np.where(values >= threshold, values, 0.0).mean())


def _bio_shark_score(first: str, second: str, *, k: int, threshold: float) -> float | None:
    if not shark_available():
        return None
    try:
        from bio_shark.dive import run  # noqa: PLC0415

        return float(run.run_normal(sequence1=first, sequence2=second, k=k, threshold=threshold))
    except Exception:  # noqa: BLE001 - third-party backend: degrade instead of aborting the run
        logger.warning("bio_shark scoring failed; falling back to BLOSUM k-mer scores.", exc_info=True)
        return None


def shark_score(
    first: str,
    second: str,
    *,
    k: int = DEFAULT_SHARK_K,
    threshold: float = DEFAULT_SHARK_THRESHOLD,
    backend: str = BACKEND_AUTO,
) -> float:
    """Score similarity between two unalignable regions.

    Args:
        first: First sequence.
        second: Second sequence.
        k: k-mer length; clamped to the shorter sequence (SHARK raises when k is larger).
        threshold: SHARK-score ``T`` threshold for counting a k-mer match.
        backend: ``auto``, ``bio_shark``, or ``blosum_kmer``.

    Sequences are upper-cased first: both backends look residues up in a substitution
    matrix keyed by upper-case letters, and a lower-case sequence would otherwise score
    0.0 as if it were unrelated, inflating the novelty score of a perfectly typical JDP.

    A pair containing a non-standard residue is scored with the k-mer backend even when
    ``bio_shark`` is installed; see :data:`_SHARK_ALPHABET` for why.
    """
    if not first or not second:
        return _MIN_SIMILARITY

    upper_first, upper_second = first.upper(), second.upper()
    effective_k = max(1, min(k, len(upper_first), len(upper_second)))
    scorable = is_shark_compatible(upper_first) and is_shark_compatible(upper_second)
    if scorable and backend in {BACKEND_AUTO, BACKEND_BIO_SHARK}:
        score = _bio_shark_score(upper_first, upper_second, k=effective_k, threshold=threshold)
        if score is not None:
            return score

    return kmer_shark_score(upper_first, upper_second, k=effective_k, threshold=threshold)


def dive_like_score(
    first: str,
    second: str,
    *,
    k_values: tuple[int, ...] = DIVE_K_VALUES,
    threshold: float = DEFAULT_SHARK_THRESHOLD,
    backend: str = BACKEND_AUTO,
) -> float:
    """Mean SHARK-score across several k-mer lengths (a SHARK-dive-style aggregate)."""
    scores = [shark_score(first, second, k=k, threshold=threshold, backend=backend) for k in k_values]
    if not scores:
        return _MIN_SIMILARITY
    return sum(scores) / len(scores)


def best_match(
    sequence: str,
    references: Mapping[str, str],
    *,
    k_values: tuple[int, ...] = DIVE_K_VALUES,
    threshold: float = DEFAULT_SHARK_THRESHOLD,
    backend: str = BACKEND_AUTO,
) -> SharkMatch | None:
    """Return the highest-scoring reference region for one query region."""
    if not sequence or not references:
        return None

    best: SharkMatch | None = None
    for reference_id in sorted(references):
        reference = references[reference_id]
        score = dive_like_score(
            sequence,
            reference,
            k_values=k_values,
            threshold=threshold,
            backend=backend,
        )
        if best is None or score > best.score:
            best = SharkMatch(
                reference_id=reference_id,
                score=score,
                # Reported per pair, not per run: a region with a non-standard residue is
                # scored by the k-mer backend even on a bio_shark run.
                backend=resolved_backend_for(sequence, reference, backend=backend),
            )
    return best
