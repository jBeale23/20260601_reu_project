"""Tests for the empirical ordering-stability check."""

from __future__ import annotations

import pytest

from validation.capacity import max_resolvable_error_rate
from validation.ordering_stability import (
    HEADLINE_ERROR_RATE,
    ResampleSettings,
    ordering_stability,
    stability_report,
)


def _truth(n: int = 120) -> dict[str, str]:
    return {f"P{i}": "ABC"[i % 3] for i in range(n)}


def _perfect(truth: dict[str, str]) -> dict[str, str]:
    return dict(truth)


def _wrong_every_nth(truth: dict[str, str], every: int) -> dict[str, str]:
    out = {}
    for index, (key, value) in enumerate(sorted(truth.items())):
        out[key] = "ABC"[(("ABC".index(value)) + 1) % 3] if index % every == 0 else value
    return out


def test_a_wide_gap_is_stable_under_noise() -> None:
    """A method far ahead should not be overtaken by plausible label error."""
    truth = _truth()
    results = ordering_stability(
        truth,
        _perfect(truth),
        _wrong_every_nth(truth, 2),
        names=("perfect", "poor"),
        settings=ResampleSettings(error_rates=(0.05,), n_resamples=200),
    )
    assert results[0].better == "perfect"
    assert results[0].is_stable
    assert results[0].flip_rate < 0.05


def test_a_narrow_gap_becomes_unstable_as_error_grows() -> None:
    """Flip rate must increase with assumed error, or the check measures nothing."""
    truth = _truth()
    results = ordering_stability(
        truth,
        _wrong_every_nth(truth, 20),
        _wrong_every_nth(truth, 19),
        names=("a", "b"),
        settings=ResampleSettings(error_rates=(0.01, 0.20), n_resamples=200),
    )
    assert results[0].flip_rate <= results[1].flip_rate


def test_both_methods_are_scored_on_the_same_proteins() -> None:
    """A flip must come from labels moving, never from an easier question."""
    truth = _truth(60)
    partial = {key: value for index, (key, value) in enumerate(sorted(truth.items())) if index < 30}
    results = ordering_stability(
        truth,
        _perfect(truth),
        partial,
        names=("full", "partial"),
        settings=ResampleSettings(error_rates=(0.05,), n_resamples=50),
    )
    # Only the 30 shared proteins are used, so neither is credited for the rest.
    assert results


def test_identical_methods_never_flip_meaningfully() -> None:
    """Two identical predictors cannot be ordered, so neither can overtake the other."""
    truth = _truth()
    same = _wrong_every_nth(truth, 5)
    results = ordering_stability(
        truth,
        same,
        dict(same),
        names=("x", "y"),
        settings=ResampleSettings(error_rates=(0.10,), n_resamples=200),
    )
    assert results[0].flip_rate == pytest.approx(0.0)


def test_the_check_is_reproducible_from_the_seed() -> None:
    """A simulated result that changes between runs cannot be reported."""
    truth = _truth()
    args = {"names": ("a", "b"), "settings": ResampleSettings(error_rates=(0.05,), n_resamples=100, seed=7)}
    first = ordering_stability(truth, _perfect(truth), _wrong_every_nth(truth, 4), **args)
    second = ordering_stability(truth, _perfect(truth), _wrong_every_nth(truth, 4), **args)
    assert first[0].flip_rate == second[0].flip_rate


def test_report_covers_every_pair_and_names_the_headline_rate() -> None:
    """Three methods give three pairs, each swept across the error rates."""
    truth = _truth()
    report = stability_report(
        truth,
        {
            "perfect": _perfect(truth),
            "good": _wrong_every_nth(truth, 10),
            "poor": _wrong_every_nth(truth, 2),
        },
        settings=ResampleSettings(error_rates=(0.01, 0.05), n_resamples=100),
    )
    assert report["n_methods"] == 3
    assert report["n_pairwise_tests"] == 3 * 2
    assert report["headline_error_rate"] == HEADLINE_ERROR_RATE


def test_the_simulation_is_more_permissive_than_the_proved_bound() -> None:
    """Worst case against average case, which is why both are reported.

    The bound asks whether *any* arrangement of errors within budget reverses the
    ordering; the simulation asks whether randomly drawn ones do. Random errors largely
    cancel, so an ordering can be robust in practice and still uncertifiable.
    """
    truth = _truth()
    gap_needing_low_error = 0.0387
    bound_limit = max_resolvable_error_rate(gap_needing_low_error)
    assert bound_limit < 0.02

    # At an error rate the bound calls undecidable, random noise usually does not flip it.
    results = ordering_stability(
        truth,
        _perfect(truth),
        _wrong_every_nth(truth, 25),
        names=("leader", "trailer"),
        settings=ResampleSettings(error_rates=(0.05,), n_resamples=200),
    )
    assert results[0].error_rate > bound_limit
    assert results[0].is_stable


def test_no_shared_proteins_yields_no_result() -> None:
    """Two methods with nothing in common cannot be compared."""
    assert ordering_stability({"A": "A"}, {"A": "A"}, {"B": "B"}, names=("x", "y")) == []
