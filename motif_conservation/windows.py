"""Sliding-window charge profiles and conservation scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from motif_conservation.charge_alphabet import charge_profile, sequence_net_charge
from motif_conservation.constants import DEFAULT_WINDOW_LENGTHS

if TYPE_CHECKING:
    from motif_conservation.alignment_frames import AlignedFamily

_MIN_HOMOLOGS_FOR_CONSERVATION = 2


@dataclass(frozen=True, slots=True)
class WindowSweepResult:
    """Conservation-vs-length sweep for one aligned family."""

    domain_family: str
    n_members: int
    n_aligned_columns: int
    scores_by_length: dict[int, float]
    mean_charge_by_length: dict[int, float]
    best_window_length: int
    best_conservation_score: float


def column_charge_matrix(aligned_sequences: tuple[str, ...] | list[str]) -> list[list[int]]:
    """Return per-sequence charge profiles over aligned columns."""
    return [charge_profile(seq) for seq in aligned_sequences]


def window_net_charges(profile: list[int], window: int) -> list[float]:
    """Sliding-window mean net charge along one profile."""
    if window <= 0 or window > len(profile):
        return []
    return [sum(profile[i : i + window]) / window for i in range(len(profile) - window + 1)]


def conservation_score_for_window(matrix: list[list[int]], window: int) -> tuple[float, float]:
    """Score cross-homolog conservation of sliding-window charge profiles.

    Higher is better: informative shared pattern with low across-homolog variance.
    Returns (conservation_score, mean_abs_window_charge).
    """
    if not matrix or window <= 0:
        return 0.0, 0.0

    profiles = [window_net_charges(row, window) for row in matrix]
    profiles = [row for row in profiles if row]
    if len(profiles) < _MIN_HOMOLOGS_FOR_CONSERVATION:
        return 0.0, 0.0

    length = min(len(row) for row in profiles)
    if length == 0:
        return 0.0, 0.0

    # Truncate to common length.
    profiles = [row[:length] for row in profiles]
    n = len(profiles)
    total_var = 0.0
    total_signal = 0.0
    for col in range(length):
        values = [row[col] for row in profiles]
        mean = sum(values) / n
        var = sum((value - mean) ** 2 for value in values) / n
        total_var += var
        total_signal += abs(mean)

    mean_var = total_var / length
    mean_signal = total_signal / length
    # Prefer shared nonzero charge patterns: signal / (1 + variance).
    score = mean_signal / (1.0 + mean_var)
    return score, mean_signal


def sweep_window_lengths(
    aligned: AlignedFamily,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
) -> WindowSweepResult:
    """Sweep window lengths and pick the most conserved charge scale."""
    matrix = column_charge_matrix(aligned.aligned_sequences)
    n_cols = len(aligned.aligned_sequences[0]) if aligned.aligned_sequences else 0
    scores: dict[int, float] = {}
    means: dict[int, float] = {}
    for window in window_lengths:
        if window > n_cols:
            continue
        score, mean_charge = conservation_score_for_window(matrix, window)
        scores[window] = score
        means[window] = mean_charge

    if not scores:
        return WindowSweepResult(
            domain_family=aligned.domain_family,
            n_members=len(aligned.accessions),
            n_aligned_columns=n_cols,
            scores_by_length={},
            mean_charge_by_length={},
            best_window_length=0,
            best_conservation_score=0.0,
        )

    best_window = max(scores, key=lambda length: (scores[length], -length))
    return WindowSweepResult(
        domain_family=aligned.domain_family,
        n_members=len(aligned.accessions),
        n_aligned_columns=n_cols,
        scores_by_length=scores,
        mean_charge_by_length=means,
        best_window_length=best_window,
        best_conservation_score=scores[best_window],
    )


def per_sequence_window_charge(sequence: str, window: int) -> float | None:
    """Net charge of the first full window (or whole sequence if shorter)."""
    if not sequence:
        return None
    if len(sequence) < window:
        return float(sequence_net_charge(sequence))
    return float(sequence_net_charge(sequence[:window]))
