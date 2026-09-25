"""Tests for whether a finer classification is better or merely finer.

Every accuracy figure elsewhere in this project is scored against gold labels that *are*
the A/B/C subfamily, so none of them can answer this. These tests pin the experiment that
can, and in particular pin that it can return a negative - a refinement test that always
approves is not a test.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from validation.class_sufficiency import (
    MIN_PROTEINS,
    _shape_matched_shuffle,
    assess_sufficiency,
    cramers_v,
    cramers_v_corrected,
    sufficiency_report,
)


def _labels(n: int = 300) -> tuple[dict[str, str], dict[str, str]]:
    coarse = {f"P{i}": "ABC"[i % 3] for i in range(n)}
    fine = {f"P{i}": f"{coarse[f'P{i}']}{1 if i % 6 < 3 else 2}" for i in range(n)}
    return coarse, fine


def test_a_refinement_carrying_real_information_is_approved() -> None:
    """The positive case: subgroups differ in the readout, parents do not."""
    coarse, fine = _labels()
    readout = {key: ("X" if value.endswith("1") else "Y") for key, value in fine.items()}
    result = assess_sufficiency(coarse, fine, readout, n_permutations=200)
    assert result is not None
    assert result.fine_v > result.coarse_v
    assert result.beats_random_refinement


def test_an_arbitrary_refinement_is_rejected() -> None:
    """The control that makes this a test.

    Splitting three classes into six improves almost any association, for the same reason
    adding parameters improves almost any fit. A split with no content must not pass.
    """
    rng = random.Random(0)  # noqa: S311 - deterministic test fixture
    coarse, _ = _labels()
    arbitrary = {key: f"{value}{rng.choice('12')}" for key, value in coarse.items()}
    readout = {key: ("X" if int(key[1:]) % 2 else "Y") for key in coarse}
    result = assess_sufficiency(coarse, arbitrary, readout, n_permutations=200)
    assert result is not None
    assert not result.beats_random_refinement


def test_the_null_preserves_subgroup_sizes() -> None:
    """Only labels move; counts and sizes are held fixed.

    A null that changed the shape would compare granularity against granularity and answer
    a different question.
    """
    coarse, fine = _labels(60)
    rng = random.Random(1)  # noqa: S311 - deterministic test fixture
    shuffled = _shape_matched_shuffle(coarse, fine, rng)
    assert Counter(shuffled.values()) == Counter(fine.values())
    # And every protein keeps a label belonging to its own parent class.
    assert all(value.startswith(coarse[key]) for key, value in shuffled.items())


def test_cramers_v_is_zero_when_there_is_no_association() -> None:
    """A readout independent of the labelling must score zero, not a small positive."""
    pairs = [(f"C{i % 3}", f"R{i % 2}") for i in range(600)]
    assert cramers_v(pairs) < 0.05


def test_cramers_v_is_one_for_a_perfect_association() -> None:
    """A labelling that determines the readout scores one."""
    pairs = [(f"C{i % 3}", f"R{i % 3}") for i in range(300)]
    assert cramers_v(pairs) == pytest.approx(1.0)


def test_a_single_category_has_no_association() -> None:
    """One class cannot explain anything; the answer is zero, not a divide by zero."""
    assert cramers_v([("only", f"R{i % 2}") for i in range(100)]) == 0.0


def test_too_few_proteins_yields_no_result() -> None:
    """A contingency table over a handful of proteins is not a measurement."""
    coarse, fine = _labels(10)
    readout = dict.fromkeys(coarse, "X")
    assert assess_sufficiency(coarse, fine, readout, n_permutations=50) is None
    assert MIN_PROTEINS >= 30


def test_the_report_states_why_accuracy_cannot_answer_this() -> None:
    """The gold labels are A/B/C, so predicting them well says nothing about sufficiency."""
    coarse, fine = _labels()
    readout = {key: ("X" if value.endswith("1") else "Y") for key, value in fine.items()}
    report = sufficiency_report(coarse, fine, {"partner": readout}, n_permutations=100)
    assert report["n_coarse_classes"] == 3
    assert report["n_fine_classes"] == 6
    assert "gold labels are the A/B/C subfamily" in str(report["interpretation"])


def test_corrected_v_matches_plain_v_on_a_strong_two_by_two() -> None:
    """With a large sample and a real association the correction changes little."""
    pairs = [("a", "x")] * 500 + [("b", "y")] * 500
    assert cramers_v_corrected(pairs) == pytest.approx(cramers_v(pairs), abs=0.02)


def test_corrected_v_is_zero_under_independence() -> None:
    """Independent labellings must score zero, not the small positive value chance gives."""
    rng = random.Random(17)  # noqa: S311 - synthetic fixture, not a security boundary
    # Randomly assigned, so any association is sampling noise. A deterministic interleaving
    # would give exactly zero chi-square and so would not exercise the correction at all.
    pairs = [(f"r{rng.randrange(4)}", f"c{rng.randrange(5)}") for _ in range(600)]
    assert cramers_v_corrected(pairs) == 0.0
    # The uncorrected statistic is positive here, which is exactly the bias being removed.
    assert cramers_v(pairs) > 0.0


def test_correction_does_not_rescue_a_noise_refinement() -> None:
    """The correction removes the null bias; it does not make granularities comparable.

    Recorded as a test because the opposite is an easy and costly assumption: a refinement
    carrying no new information still scores below its parent under both forms, so neither
    statistic can answer "does this combination of features beat a single one". Mutual
    information is the instrument for that, and this test pins the limitation in place.
    """
    rng = random.Random(5)  # noqa: S311 - synthetic fixture, not a security boundary
    outcomes = [f"o{i}" for i in range(19)]
    coarse_pairs, fine_pairs = [], []
    for i in range(4000):
        parent = "a" if i % 2 else "b"
        outcome = outcomes[rng.randrange(0, 10)] if parent == "a" else outcomes[rng.randrange(9, 19)]
        # The refinement splits each parent group at random: no new information at all.
        child = f"{parent}{rng.randrange(4)}"
        coarse_pairs.append((parent, outcome))
        fine_pairs.append((child, outcome))

    assert cramers_v(coarse_pairs) > cramers_v(fine_pairs)
    assert cramers_v_corrected(coarse_pairs) > cramers_v_corrected(fine_pairs)


def test_corrected_v_still_rewards_a_real_refinement() -> None:
    """A refinement that does carry information must score above its parent."""
    coarse_pairs, fine_pairs = [], []
    for i in range(4000):
        parent = "a" if i % 2 else "b"
        half = (i // 2) % 2
        # The outcome is determined by the child, so the parent can only reach half of it.
        outcome = f"{parent}{half}"
        coarse_pairs.append((parent, outcome))
        fine_pairs.append((f"{parent}{half}", outcome))
    assert cramers_v_corrected(fine_pairs) > cramers_v_corrected(coarse_pairs)


def test_corrected_v_handles_degenerate_tables() -> None:
    """One category on either side is no association, and must not divide by zero."""
    assert cramers_v_corrected([]) == 0.0
    assert cramers_v_corrected([("a", "x"), ("a", "y")]) == 0.0
    assert cramers_v_corrected([("a", "x"), ("b", "x")]) == 0.0
