"""Tests for the grammar/structure/function information split."""

from __future__ import annotations

import json
import math

import pytest

from validation.modality_information import (
    MIN_PROTEINS,
    SEQUENCE_DERIVED_PROVENANCES,
    Modality,
    ReportSettings,
    analyse_term,
    modality_report,
    mutual_information,
    profile_states,
    quantile_bins,
)


def test_mutual_information_of_identical_variables_is_the_entropy() -> None:
    """A variable predicts itself perfectly, so MI equals its entropy - here 1 bit."""
    values = [0, 1] * 500
    assert mutual_information(values, values) == pytest.approx(1.0, abs=0.01)


def test_mutual_information_of_independent_variables_is_near_zero() -> None:
    """Independent variables must not show information; this is what the correction buys."""
    first = [i % 2 for i in range(2000)]
    second = [(i // 2) % 2 for i in range(2000)]
    assert mutual_information(first, second) < 0.01


def test_mutual_information_is_never_negative() -> None:
    """The Miller-Madow correction can overshoot on sparse tables; the result is clamped."""
    first = list(range(60))
    second = [i * 7 % 60 for i in range(60)]
    assert mutual_information(first, second) >= 0.0


def test_mutual_information_handles_empty_and_mismatched_input() -> None:
    """Bad input yields zero rather than raising inside a long cluster run."""
    assert mutual_information([], []) == 0.0
    assert mutual_information([1, 2, 3], [1, 2]) == 0.0


def test_quantile_bins_split_a_skewed_variable_evenly() -> None:
    """Equal-width bins would collapse this; quantiles must not."""
    values = [0.0] * 100 + [1.0] * 10 + [50.0] * 10
    bins = quantile_bins(values, 3)
    # The mass at zero cannot be split, but the tail must land in a higher bin.
    assert max(bins) > 0
    assert bins[-1] > bins[0]


def test_profile_states_keep_features_distinguishable() -> None:
    """Two features that differ must not collapse to the same profile state."""
    features = {
        "A": {"x": 0.0, "y": 10.0},
        "B": {"x": 10.0, "y": 0.0},
        "C": {"x": 5.0, "y": 5.0},
    }
    states = profile_states(["A", "B", "C"], features, ["x", "y"])
    assert len(set(states)) == 3


def test_profile_states_tolerate_a_missing_feature() -> None:
    """A protein missing a column is binned at zero rather than crashing the run."""
    features = {"A": {"x": 1.0}, "B": {"x": 2.0}, "C": {}}
    states = profile_states(["A", "B", "C"], features, ["x"])
    assert len(states) == 3


def test_redundant_modalities_are_detected() -> None:
    """When structure is a copy of grammar, the two must read as redundant, not synergistic."""
    labels, grammar, structure = [], [], []
    for i in range(600):
        bit = i % 2
        labels.append(bit)
        grammar.append((bit,))
        structure.append((bit,))  # identical information
    result = analyse_term("t", labels, grammar, structure, n_permutations=30)
    assert result.interaction_bits > 0
    assert result.relationship == "redundant"


def test_synergistic_modalities_are_detected() -> None:
    """XOR: neither modality alone says anything, but together they determine the label."""
    labels, grammar, structure = [], [], []
    for i in range(800):
        a, b = i % 2, (i // 2) % 2
        labels.append(a ^ b)
        grammar.append((a,))
        structure.append((b,))
    result = analyse_term("t", labels, grammar, structure, n_permutations=30)
    assert result.interaction_bits < 0
    assert result.relationship == "synergistic"


def test_uninformative_modalities_are_named_as_such() -> None:
    """Profiles unrelated to the label must not be reported as a weak relationship."""
    labels = [i % 2 for i in range(600)]
    grammar = [(i % 3,) for i in range(600)]
    structure = [(i % 5,) for i in range(600)]
    result = analyse_term("t", labels, grammar, structure, n_permutations=30)
    assert result.relationship == "neither_informative"


def test_report_refuses_to_run_on_too_few_proteins() -> None:
    """Below the floor it returns an explanation, not a table nobody should trust."""
    terms = {f"A{i}": ("go:x",) for i in range(10)}
    feats = {f"A{i}": {"g": float(i), "s": float(i)} for i in range(10)}
    report = modality_report(terms, Modality(feats, ["g"]), Modality(feats, ["s"]))
    assert report["results"] == []
    assert str(MIN_PROTEINS) in report["interpretation"]


def test_report_records_provenance_verbatim() -> None:
    """Observed and transferred terms must never be conflated, so provenance is carried."""
    n = MIN_PROTEINS * 3
    terms, grammar, structure = {}, {}, {}
    for i in range(n):
        a = f"A{i}"
        terms[a] = ("go:common",) if i % 2 else ("go:common", "go:other")
        grammar[a] = {"g": float(i % 7)}
        structure[a] = {"s": float(i % 5)}
    report = modality_report(
        terms,
        Modality(grammar, ["g"]),
        Modality(structure, ["s"]),
        settings=ReportSettings(n_permutations=10, provenance="transferred"),
    )
    assert report["provenance"] == "transferred"
    if report["summary"]:
        assert "transferred by homology" in report["interpretation"]


def test_report_is_json_safe_and_counts_states() -> None:
    """The report must serialise and say how large each profile space was."""
    n = MIN_PROTEINS * 3
    terms, grammar, structure = {}, {}, {}
    for i in range(n):
        a = f"A{i}"
        terms[a] = ("go:a",) if i % 3 else ("go:a", "go:b")
        grammar[a] = {"g1": float(i % 9), "g2": float(i % 4)}
        structure[a] = {"s1": float(i % 6)}
    report = modality_report(
        terms,
        Modality(grammar, ["g1", "g2"]),
        Modality(structure, ["s1"]),
        settings=ReportSettings(n_permutations=10, provenance="observed"),
    )
    assert report["n_grammar_states"] >= 1
    assert report["n_structure_states"] >= 1
    for row in report["results"]:
        assert math.isfinite(row["interaction_bits"])
    json.dumps(report)


def test_report_skips_terms_carried_by_everyone() -> None:
    """A universal term has no variance, so it cannot be explained by anything."""
    n = MIN_PROTEINS * 2
    terms = {f"A{i}": ("go:universal",) for i in range(n)}
    grammar = {f"A{i}": {"g": float(i % 5)} for i in range(n)}
    structure = {f"A{i}": {"s": float(i % 3)} for i in range(n)}
    report = modality_report(
        terms,
        Modality(grammar, ["g"]),
        Modality(structure, ["s"]),
        settings=ReportSettings(n_permutations=5),
    )
    assert report["results"] == []


def test_sequence_derived_provenance_always_carries_its_caveat() -> None:
    """Renaming a tier in a driver must not silently drop the circularity warning."""
    n = MIN_PROTEINS * 3
    terms, grammar, structure = {}, {}, {}
    for i in range(n):
        a = f"A{i}"
        terms[a] = ("go:common",) if i % 2 else ("go:common", "go:other")
        grammar[a] = {"g": float(i % 7)}
        structure[a] = {"s": float(i % 5)}
    for provenance in SEQUENCE_DERIVED_PROVENANCES:
        report = modality_report(
            terms,
            Modality(grammar, ["g"]),
            Modality(structure, ["s"]),
            settings=ReportSettings(n_permutations=10, provenance=provenance),
        )
        if report["summary"]:
            assert "inside track" in report["interpretation"], provenance


def test_experimental_provenance_carries_no_circularity_caveat() -> None:
    """Experimental terms were not produced from sequence, so the warning must not appear."""
    n = MIN_PROTEINS * 3
    terms, grammar, structure = {}, {}, {}
    for i in range(n):
        a = f"A{i}"
        terms[a] = ("go:common",) if i % 2 else ("go:common", "go:other")
        grammar[a] = {"g": float(i % 7)}
        structure[a] = {"s": float(i % 5)}
    report = modality_report(
        terms,
        Modality(grammar, ["g"]),
        Modality(structure, ["s"]),
        settings=ReportSettings(n_permutations=10, provenance="experimental"),
    )
    assert "inside track" not in report["interpretation"]
