"""Tests for the statistical machinery: metrics, nulls, and multiple-testing control.

These check the mathematics against hand-computable cases. A validation suite whose own
statistics are wrong is worse than none, because it lends false confidence.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from validation.metrics import ClassMetrics, evaluate, majority_baseline, wilson_interval
from validation.recurrence import (
    benjamini_hochberg,
    genus_of,
    observed_recurrence,
    recurrence_with_null,
)


def test_wilson_interval_matches_known_values() -> None:
    """Wilson interval reproduces textbook values and stays inside [0, 1]."""
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.005)
    assert high == pytest.approx(0.596, abs=0.005)

    # Degenerate cases must not escape the unit interval or divide by zero.
    assert wilson_interval(0, 0) == (0.0, 0.0)
    zero_low, zero_high = wilson_interval(0, 20)
    assert zero_low == 0.0
    assert 0.0 < zero_high < 1.0
    perfect_low, perfect_high = wilson_interval(20, 20)
    assert perfect_high == 1.0
    assert 0.0 < perfect_low < 1.0


def test_wilson_interval_narrows_with_more_data() -> None:
    """More observations at the same proportion give a tighter interval."""
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_class_metrics_arithmetic() -> None:
    """Precision, recall, and F1 follow their definitions, including empty cases."""
    metrics = ClassMetrics(label="A", support=10, predicted=8, true_positives=6)
    assert metrics.precision == pytest.approx(6 / 8)
    assert metrics.recall == pytest.approx(6 / 10)
    assert metrics.f1 == pytest.approx(2 * (0.75 * 0.6) / (0.75 + 0.6))

    empty = ClassMetrics(label="B", support=0, predicted=0, true_positives=0)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def test_evaluate_builds_confusion_matrix() -> None:
    """The confusion matrix counts actual-versus-predicted pairs."""
    truth = {"p1": "A", "p2": "A", "p3": "B", "p4": "C"}
    predictions = {"p1": "A", "p2": "B", "p3": "B", "p4": "C"}

    result = evaluate("test", truth, predictions)
    assert result.n_evaluated == 4
    assert result.n_correct == 3
    assert result.accuracy == pytest.approx(0.75)
    assert result.confusion["A"]["A"] == 1
    assert result.confusion["A"]["B"] == 1
    assert result.confusion["C"]["C"] == 1


def test_evaluate_only_scores_shared_accessions() -> None:
    """A predictor is neither credited nor penalised for proteins it never saw."""
    truth = {"p1": "A", "p2": "B", "p3": "C"}
    predictions = {"p1": "A"}
    result = evaluate("partial", truth, predictions)
    assert result.n_evaluated == 1
    assert result.accuracy == 1.0


def test_macro_f1_treats_classes_equally() -> None:
    """Macro-F1 does not let a dominant class hide failure on a rare one."""
    truth = {f"p{i}": "A" for i in range(99)} | {"rare": "C"}
    predictions = dict.fromkeys(truth, "A")

    result = evaluate("majority-like", truth, predictions)
    assert result.accuracy == pytest.approx(0.99)
    # One class perfect, the other missed entirely, so macro-F1 must be far below accuracy.
    assert result.macro_f1 < 0.6


def test_majority_baseline_predicts_the_commonest_label() -> None:
    """The baseline any classifier must beat."""
    truth = {"p1": "C", "p2": "C", "p3": "A"}
    baseline = majority_baseline(truth, sorted(truth))
    assert set(baseline.values()) == {"C"}
    assert majority_baseline({}, []) == {}


def test_benjamini_hochberg_controls_false_discovery() -> None:
    """Adjusted p-values are monotone, never shrink below the raw value, and stay <= 1."""
    raw = {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5, "e": 0.9}
    adjusted = benjamini_hochberg(raw)

    assert set(adjusted) == set(raw)
    for key, value in raw.items():
        assert adjusted[key] >= value - 1e-12, key
        assert adjusted[key] <= 1.0
    ordered = sorted(raw, key=lambda k: raw[k])
    for earlier, later in pairwise(ordered):
        assert adjusted[earlier] <= adjusted[later] + 1e-12


def test_benjamini_hochberg_known_case() -> None:
    """Hand-checked: p=0.01 among 5 tests at rank 2 adjusts to 0.025."""
    adjusted = benjamini_hochberg({"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5, "e": 0.9})
    assert adjusted["b"] == pytest.approx(0.025, abs=1e-9)
    assert adjusted["e"] == pytest.approx(0.9, abs=1e-9)


def test_benjamini_hochberg_empty() -> None:
    """No tests means no adjustments."""
    assert benjamini_hochberg({}) == {}


def test_genus_extraction() -> None:
    """Genus is the first token of the organism name."""
    assert genus_of("Escherichia coli") == "Escherichia"
    assert genus_of("Oryza sativa subsp. japonica") == "Oryza"
    assert genus_of("") == "unknown"


def test_observed_recurrence_counts_taxa() -> None:
    """Recurrence counts distinct genera and organisms per architecture."""
    architectures = {"p1": "j>c", "p2": "j>c", "p3": "j>c", "p4": "j"}
    organisms = {
        "p1": "Escherichia coli",
        "p2": "Oryza sativa",
        "p3": "Escherichia coli",
        "p4": "Homo sapiens",
    }
    recurrence = observed_recurrence(architectures, organisms)
    assert recurrence["j>c"].n_proteins == 3
    assert recurrence["j>c"].n_genera == 2
    assert recurrence["j>c"].n_organisms == 2
    assert recurrence["j"].n_genera == 1


def test_recurrence_null_detects_real_taxonomic_spread() -> None:
    """An architecture spread across many genera beats its permutation null.

    Construction: one architecture appears once in each of many genera (maximal spread),
    while a second is confined to a single genus despite being equally common. Only the
    first should be significant.
    """
    architectures: dict[str, str] = {}
    organisms: dict[str, str] = {}
    for index in range(20):
        architectures[f"spread{index}"] = "j>rare_partner"
        organisms[f"spread{index}"] = f"Genus{index} species"
        architectures[f"local{index}"] = "j>common_partner"
        organisms[f"local{index}"] = "Oneus species"

    recurrence = recurrence_with_null(architectures, organisms, n_permutations=200, seed=1)
    spread = recurrence["j>rare_partner"]
    local = recurrence["j>common_partner"]

    assert spread.n_genera == 20
    assert local.n_genera == 1
    assert spread.p_value < local.p_value
    assert spread.genera_enrichment > local.genera_enrichment


def test_recurrence_null_p_value_is_never_zero() -> None:
    """A finite permutation count cannot justify p = 0."""
    architectures = {f"p{i}": "j>x" for i in range(10)}
    organisms = {f"p{i}": f"Genus{i} sp" for i in range(10)}
    recurrence = recurrence_with_null(architectures, organisms, n_permutations=50, seed=0)
    assert recurrence["j>x"].p_value > 0.0


def test_recurrence_is_reproducible_from_the_seed() -> None:
    """The same seed must give identical p-values, or results are not reproducible."""
    architectures = {f"p{i}": f"arch{i % 3}" for i in range(30)}
    organisms = {f"p{i}": f"Genus{i % 7} sp" for i in range(30)}
    first = recurrence_with_null(architectures, organisms, n_permutations=100, seed=42)
    second = recurrence_with_null(architectures, organisms, n_permutations=100, seed=42)
    assert {k: v.p_value for k, v in first.items()} == {k: v.p_value for k, v in second.items()}


def test_recurrence_null_is_calibrated_on_random_data() -> None:
    """With no real association, p-values must not be systematically small.

    This is the honesty check on the null itself: assign architectures to organisms at
    random and confirm the test does not manufacture significance.
    """
    rng = random.Random(7)  # noqa: S311 - deterministic test fixture
    architectures = {f"p{i}": f"arch{rng.randrange(4)}" for i in range(200)}
    organisms = {f"p{i}": f"Genus{rng.randrange(25)} sp" for i in range(200)}

    recurrence = recurrence_with_null(architectures, organisms, n_permutations=200, seed=3)
    significant = [item for item in recurrence.values() if item.p_value < 0.05]
    # At most a small fraction of the four architectures may look significant by chance.
    assert len(significant) <= 1
