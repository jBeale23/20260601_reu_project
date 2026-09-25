"""Tests for what the sequence grammar tracks.

The grammar model was the only one that improved when rare architectures were added, so it
is worth knowing what it reads. The danger is that grammar features and the things they are
correlated against are both computed from the same sequence, which makes tautologies easy
to mistake for findings. These tests pin the controls that separate the two.
"""

from __future__ import annotations

import math
import random

import pytest

from validation.grammar_correlates import (
    DEFAULT_CONTROLS,
    MIN_PAIRED_OBSERVATIONS,
    Correlate,
    correlate_grammar,
    grammar_correlate_report,
    partial_spearman,
    spearman,
)


def test_a_perfect_monotone_relationship_is_one() -> None:
    """Rank correlation, so any increasing transform must give the same answer."""
    xs = list(range(30))
    assert spearman(xs, [x**3 for x in xs]) == pytest.approx(1.0)
    assert spearman(xs, [-x for x in xs]) == pytest.approx(-1.0)


def test_ties_do_not_bias_the_correlation() -> None:
    """Grammar order scores cluster hard near zero, so ties are the normal case."""
    xs = [1, 1, 1, 2, 2, 3]
    assert abs(spearman(xs, [1, 1, 1, 2, 2, 3])) == pytest.approx(1.0)


def test_an_association_driven_entirely_by_a_confounder_vanishes() -> None:
    """The whole point of partialling.

    If two measures are correlated only because both track protein length, the partial
    correlation must go to zero - otherwise every long protein's every property would read
    as a grammar finding.
    """
    rng = random.Random(0)  # noqa: S311 - deterministic test fixture
    length = [rng.uniform(50, 2000) for _ in range(200)]
    # Both track length, with noise large enough that neither is a perfect copy of it -
    # a perfect copy makes the partialling denominator singular, which is a separate case.
    a = [item + rng.gauss(0, 400) for item in length]
    b = [item + rng.gauss(0, 400) for item in length]
    assert spearman(a, b) > 0.6
    assert abs(partial_spearman(a, b, [length])) < 0.3


def test_a_real_association_survives_partialling() -> None:
    """The control must not erase relationships that are not about the confounder."""
    rng = random.Random(1)  # noqa: S311 - deterministic test fixture
    length = [rng.uniform(50, 2000) for _ in range(200)]
    signal = [rng.gauss(0, 1) for _ in range(200)]
    a = [s + 0.01 * item for s, item in zip(signal, length, strict=True)]
    b = [s + 0.01 * item for s, item in zip(signal, length, strict=True)]
    assert abs(partial_spearman(a, b, [length])) > 0.5


def test_too_few_paired_observations_are_not_correlated() -> None:
    """A correlation over a handful of proteins is noise with a decimal point."""
    grammar = {f"P{i}": {"entropy": float(i)} for i in range(5)}
    other = {f"P{i}": {"n_pockets": float(i)} for i in range(5)}
    assert correlate_grammar(grammar, other) == []
    assert MIN_PAIRED_OBSERVATIONS >= 20


def test_a_control_is_never_correlated_against_itself() -> None:
    """Otherwise every report would lead with a perfect self-correlation."""
    rng = random.Random(2)  # noqa: S311 - deterministic test fixture
    grammar = {f"P{i}": {"entropy": rng.random()} for i in range(60)}
    other = {
        f"P{i}": {"n_residues": rng.random(), "fraction_disordered_plddt": rng.random(), "n_pockets": rng.random()}
        for i in range(60)
    }
    results = correlate_grammar(grammar, other)
    assert all(r.against not in DEFAULT_CONTROLS for r in results)
    assert any(r.against == "n_pockets" for r in results)


def test_surviving_controls_requires_both_magnitude_and_retention() -> None:
    """A halved association was largely the confounder; a tiny one says nothing."""
    halved = Correlate("g", "x", n=100, rho=0.8, partial_rho=0.2, p_value=0.0)
    assert halved.survives_controls is False
    tiny = Correlate("g", "x", n=100, rho=0.08, partial_rho=0.08, p_value=0.0)
    assert tiny.survives_controls is False
    real = Correlate("g", "x", n=100, rho=0.6, partial_rho=0.5, p_value=0.0)
    assert real.survives_controls is True


def test_the_report_states_that_nothing_here_is_causal() -> None:
    """The distinction is the difference between a finding and an overclaim."""
    rng = random.Random(3)  # noqa: S311 - deterministic test fixture
    grammar = {f"P{i}": {"entropy": rng.random()} for i in range(40)}
    other = {
        f"P{i}": {"n_residues": rng.random(), "fraction_disordered_plddt": rng.random(), "n_pockets": rng.random()}
        for i in range(40)
    }
    report = grammar_correlate_report(grammar, other)
    assert "causal" in str(report["interpretation"])
    assert report["controls"] == list(DEFAULT_CONTROLS)


def test_a_constant_column_does_not_produce_a_spurious_correlation() -> None:
    """Zero variance must give zero, not a division by zero or a NaN."""
    xs = [1.0] * 40
    ys = [float(i) for i in range(40)]
    assert spearman(xs, ys) == 0.0
    assert not math.isnan(spearman(xs, ys))


def test_a_control_that_explains_everything_leaves_no_residual() -> None:
    """A singular partialling denominator must give zero, not an arbitrary ratio.

    When a control is a near-perfect copy of a variable there is nothing left to partial;
    the division becomes unstable and produced a spurious 0.36 before this was guarded.
    """
    length = [float(i) for i in range(200)]
    exact = list(length)
    assert spearman(exact, length) == pytest.approx(1.0)
    assert partial_spearman(exact, exact, [length]) == 0.0
