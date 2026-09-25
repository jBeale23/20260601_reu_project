"""Tests for the benchmark-capacity bound.

The bound says how many methods a noisy label set can place in a certified order. These
check the arithmetic against hand-computable cases and pin the property that motivates
reporting it at all: more targets do not buy resolution the labels lack.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from validation.capacity import (
    REFERENCE_ERROR_RATES,
    benchmark_capacity,
    capacity_ceiling,
    capacity_report,
    max_resolvable_error_rate,
    pairwise_decidability,
    sharp_decidability,
    sharp_decidability_report,
)


def test_capacity_matches_the_formula_on_hand_computed_cases() -> None:
    """k(n, eps) = n // (floor(2*eps*n) + 1) + 1."""
    # n=129, eps=0.05 -> c=0.10, floor(12.9)=12, 129//13 + 1 = 9 + 1 = 10
    assert benchmark_capacity(129, 0.05) == 10
    # n=39, eps=0.10 -> c=0.20, floor(7.8)=7, 39//8 + 1 = 4 + 1 = 5
    assert benchmark_capacity(39, 0.10) == 5


def test_more_targets_do_not_buy_resolution_the_labels_lack() -> None:
    """The result that makes this worth reporting.

    With delta = 0 the capacity is bounded by ceil(1 / (2*eps)) whatever n is, so a
    benchmark cannot be rescued by collecting more proteins - only by better labels.
    """
    ceiling = capacity_ceiling(0.08)
    for n in (100, 1_000, 10_000, 1_000_000):
        assert benchmark_capacity(n, 0.08) <= ceiling


def test_capacity_falls_as_labels_get_noisier() -> None:
    """Monotone in the error rate, or the bound would not mean what it says."""
    capacities = [benchmark_capacity(129, rate) for rate in REFERENCE_ERROR_RATES]
    for higher, lower in pairwise(capacities):
        assert higher >= lower


def test_perfect_labels_impose_no_ceiling() -> None:
    """With no noise and no minimum difference, nothing is undecidable."""
    assert benchmark_capacity(50, 0.0) == 50
    assert capacity_ceiling(0.0) == 0


def test_degenerate_sizes_do_not_crash() -> None:
    """An empty benchmark orders exactly one method: none of them against nothing."""
    assert benchmark_capacity(0, 0.05) == 1
    assert benchmark_capacity(-3, 0.05) == 1


def test_a_gap_is_decidable_only_below_half_its_size() -> None:
    """Two-sided noise: a gap survives only while 2*eps stays under it."""
    assert max_resolvable_error_rate(0.04) == pytest.approx(0.02)
    assert max_resolvable_error_rate(0.0) == 0.0
    # Negative gaps are not meaningful and must not produce a negative threshold.
    assert max_resolvable_error_rate(-0.1) == 0.0


def test_the_projects_own_comparison_is_reported_as_marginal() -> None:
    """The layout classifier against the profile HMM, at the scores actually measured.

    0.9612 - 0.9225 = 0.0387, so the ordering needs labels better than ~1.9% error. That
    is the number the report has to state rather than presenting the ranking as settled.
    """
    evaluations = [
        {"name": "domain_layout", "accuracy": 0.9225},
        {"name": "profile_hmm_cv", "accuracy": 0.9612},
    ]
    pairs = pairwise_decidability(evaluations)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["better"] == "profile_hmm_cv"
    assert pair["max_error_rate_for_a_decidable_ordering"] == pytest.approx(0.0194, abs=1e-3)
    assert pair["decidable_at"]["1%"] is True
    assert pair["decidable_at"]["5%"] is False


def test_pairs_are_ordered_closest_first() -> None:
    """The least defensible comparison should be the first one a reader meets."""
    evaluations = [
        {"name": "a", "accuracy": 0.90},
        {"name": "b", "accuracy": 0.91},
        {"name": "c", "accuracy": 0.50},
    ]
    gaps = [pair["score_gap"] for pair in pairwise_decidability(evaluations)]
    assert gaps == sorted(gaps)


def test_report_flags_when_capacity_is_below_the_number_of_methods() -> None:
    """The whole point: say when the ranking asks more of the labels than they hold."""
    evaluations = [{"name": f"m{index}", "accuracy": 0.9 - index * 0.001} for index in range(6)]
    report = capacity_report(129, evaluations, n_effective=39)

    assert report["n_methods_compared"] == 6
    assert report["n_effective_targets"] == 39
    # At 15% label error the capacity is 4, which cannot order six methods.
    assert report["capacity_by_error_rate"]["15%"]["sufficient_for_this_comparison"] is False
    # At 1% it comfortably can.
    assert report["capacity_by_error_rate"]["1%"]["sufficient_for_this_comparison"] is True


def test_report_defaults_effective_targets_to_the_raw_count() -> None:
    """Without an independence estimate the report must not invent one."""
    report = capacity_report(50, [{"name": "a", "accuracy": 0.9}])
    assert report["n_effective_targets"] == 50


# The sharp per-pair test
# -----------------------


def test_shared_failures_cannot_reverse_an_ordering() -> None:
    """Targets both methods get wrong move both scores together, so they are not usable.

    The crude bound would charge the full error budget against the margin; the sharp form
    charges only the targets where the loser fails and the winner does not.
    """
    truth = {f"P{i}": "A" for i in range(100)}
    # Both wrong on the same 30 targets; second additionally wrong on 10 more.
    first = {key: ("B" if index < 30 else "A") for index, key in enumerate(sorted(truth))}
    second = {key: ("B" if index < 40 else "A") for index, key in enumerate(sorted(truth))}

    result = sharp_decidability(truth, first, second, names=("first", "second"), error_rate=0.05)
    assert result is not None
    assert result["better"] == "first"
    assert result["margin_targets"] == 10
    # Only the 10 exclusive failures are exploitable, not all 40 of second's errors.
    assert result["exploitable_targets"] == 10
    # Budget is 5 targets, so at most 10 can be swung - exactly the margin, which holds.
    assert result["error_budget_targets"] == 5
    assert result["decidable"] is True


def test_the_sharp_test_is_never_looser_than_the_crude_one() -> None:
    """The sharp form agrees with the crude bound when every failure is exclusive.

    It certifies more pairs only by declining to charge shared failures, never by
    weakening the test itself.
    """
    truth = {f"P{i}": "AB"[i % 2] for i in range(100)}
    first = dict(truth)
    second = {key: ("A" if index < 8 else truth[key]) for index, key in enumerate(sorted(truth))}
    sharp = sharp_decidability(truth, first, second, names=("a", "b"), error_rate=0.05)
    assert sharp is not None
    # Every one of second's failures is exclusive here, so the sharp form charges the full
    # budget and agrees with the crude bound.
    assert sharp["exploitable_targets"] == sharp["margin_targets"]


def test_equal_scores_have_no_ordering_to_defend() -> None:
    """Two methods that score identically are not ranked, so nothing can be reversed."""
    truth = {f"P{i}": "A" for i in range(10)}
    same = dict.fromkeys(truth, "B")
    assert sharp_decidability(truth, same, dict(same), names=("a", "b"), error_rate=0.05) is None


def test_a_narrow_margin_against_exclusive_failures_is_undecidable() -> None:
    """When the loser's failures are all exclusive, a small margin dissolves."""
    truth = {f"P{i}": "A" for i in range(100)}
    first = dict(truth)
    second = {key: ("B" if index < 3 else "A") for index, key in enumerate(sorted(truth))}
    result = sharp_decidability(truth, first, second, names=("a", "b"), error_rate=0.05)
    assert result is not None
    assert result["margin_targets"] == 3
    assert result["reversible_by"] == 6
    assert result["decidable"] is False


def test_report_ranks_the_closest_pairs_first() -> None:
    """The least defensible comparison is where a reader's scepticism belongs."""
    truth = {f"P{i}": "A" for i in range(60)}
    report = sharp_decidability_report(
        truth,
        {
            "perfect": dict(truth),
            "near": {key: ("B" if i < 2 else "A") for i, key in enumerate(sorted(truth))},
            "far": {key: ("B" if i < 30 else "A") for i, key in enumerate(sorted(truth))},
        },
        error_rate=0.05,
    )
    assert report["n_pairs"] == 3
    margins = [int(pair["margin_targets"]) for pair in report["pairs"]]
    assert margins == sorted(margins)


def test_the_sharp_form_is_equivalent_to_the_crude_bound() -> None:
    """The tighter-looking hypothesis certifies exactly what ``2 * eps`` certifies.

    Worth pinning: the sharp form *looks* like it should rescue narrow margins by
    declining to charge shared failures. It cannot, because the margin never exceeds the
    exploitable count. If this ever diverges, one of the two is wrong.
    """
    rng = random.Random(20260822)  # noqa: S311 - deterministic test fixture
    n_targets = 129
    truth = {f"P{i}": "AB"[i % 2] for i in range(n_targets)}
    for _ in range(2000):
        rate_a, rate_b = rng.choice((0.05, 0.15, 0.3)), rng.choice((0.05, 0.15, 0.3))
        first = {key: ("C" if rng.random() < rate_a else value) for key, value in truth.items()}
        second = {key: ("C" if rng.random() < rate_b else value) for key, value in truth.items()}
        result = sharp_decidability(truth, first, second, names=("a", "b"), error_rate=0.05)
        if result is None:
            continue
        margin = int(result["margin_targets"])
        budget = int(result["error_budget_targets"])
        assert margin <= int(result["exploitable_targets"])
        assert result["decidable"] is (margin >= 2 * budget)
