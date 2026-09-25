"""Tests for whether J-domain protein class predicts which organ system fails.

The claim under test is strong: that a structural definition of a chaperone class predicts
pathology. These tests exist mostly to stop it being reached for the wrong reasons - a
substring collision, a singleton class, or a disease forced into one bucket when it fails
two organs.
"""

from __future__ import annotations

import pytest

from data_fetching.fetch_function import FunctionalRecord
from validation.disease import (
    MIN_DISEASE_PROTEINS_PER_CLASS,
    SIGNIFICANCE_LEVEL,
    UNGROUPED,
    disease_associations,
    disease_groups,
    disease_report,
    groups_for,
)


def _record(accession: str, *diseases: str) -> FunctionalRecord:
    return FunctionalRecord(accession=accession, diseases=tuple(diseases), is_reviewed=True)


# Grouping
# --------


def test_cardiomyopathy_is_not_filed_as_skeletal_myopathy() -> None:
    """The word myopathy is a substring of cardiomyopathy.

    A plain containment test filed every dilated cardiomyopathy under neuromuscular, which
    would have manufactured a muscle association for a class of cardiac proteins.
    """
    assert disease_groups("Hypertrophic cardiomyopathy 1") == {"cardiac"}
    assert "neuromuscular" not in disease_groups("Dilated cardiomyopathy 1A")


def test_skeletal_myopathy_is_still_caught() -> None:
    """The guard must not overshoot into missing real myopathies."""
    assert disease_groups("Myopathy, distal, 1") == {"neuromuscular"}


def test_a_multi_organ_disease_belongs_to_every_group_it_fails() -> None:
    """Forcing these into one bucket discards what makes them interesting.

    DNAJC19's syndrome fails a heart and a cerebellum; that is the observation, not noise
    to be resolved by picking a winner.
    """
    assert disease_groups("Dilated cardiomyopathy with ataxia") == {"cardiac", "neurodegenerative"}


def test_grouping_does_not_depend_on_dictionary_order() -> None:
    """First-match-wins made the result sensitive to how the table happened to be written."""
    both = disease_groups("Cardiomyopathy with retinitis pigmentosa")
    assert both == {"cardiac", "ophthalmic"}


def test_an_unrecognised_disease_is_grouped_as_other_not_dropped() -> None:
    """Silently dropping it would shrink the denominator and inflate every rate."""
    assert disease_groups("Entirely novel syndrome") == {UNGROUPED}


def test_a_protein_carries_the_union_of_its_diseases_groups() -> None:
    """Several diseases per protein is common and all of them count."""
    record = _record("P1", "Myopathy, distal, 1", "Polycystic kidney disease 6")
    assert groups_for(record) == {"neuromuscular", "renal"}


# Association
# -----------


def test_a_class_failing_one_organ_system_is_detected() -> None:
    """The signal the module exists to find."""
    classes = {f"M{i}": "muscle_class" for i in range(6)}
    classes.update({f"N{i}": "neuro_class" for i in range(6)})
    records = {f"M{i}": _record(f"M{i}", "Muscular dystrophy, limb-girdle 1") for i in range(6)}
    records.update({f"N{i}": _record(f"N{i}", "Parkinson disease 19") for i in range(6)})

    results = disease_associations(classes, records)
    muscle = [r for r in results if r.protein_class == "muscle_class" and r.disease_group == "neuromuscular"]
    assert muscle
    assert muscle[0].rate_in_class == pytest.approx(1.0)
    assert muscle[0].rate_in_others == pytest.approx(0.0)
    assert muscle[0].p_adjusted < SIGNIFICANCE_LEVEL


def test_a_class_with_too_few_disease_links_is_not_quoted() -> None:
    """One protein moves a rate by tens of points; that is not a measurement."""
    classes = {"A1": "tiny_class", "B1": "big", "B2": "big", "B3": "big", "B4": "big"}
    records = {a: _record(a, "Parkinson disease 19") for a in classes}
    results = disease_associations(classes, records)
    assert all(r.protein_class != "tiny_class" for r in results)
    assert MIN_DISEASE_PROTEINS_PER_CLASS >= 3


def test_proteins_without_a_disease_are_excluded_from_the_denominator() -> None:
    """The question is which disease a class causes, not how many members are studied."""
    classes = {f"P{i}": "c" for i in range(6)}
    records = {f"P{i}": _record(f"P{i}", "Parkinson disease 19") for i in range(3)}
    records.update({f"P{i}": FunctionalRecord(accession=f"P{i}", is_reviewed=True) for i in range(3, 6)})
    report = disease_report(classes, records)
    assert report["n_with_disease_association"] == 3
    assert report["n_classified_with_annotation"] == 6


def test_the_report_leads_with_coverage() -> None:
    """A result from a handful of human proteins is a different claim from one from many."""
    classes = {f"P{i}": "c" for i in range(4)}
    records = {f"P{i}": _record(f"P{i}", "Myopathy, distal, 1") for i in range(4)}
    report = disease_report(classes, records)
    for key in ("n_classified_with_annotation", "n_reviewed", "n_with_disease_association"):
        assert key in report
    assert "clinical study effort" in str(report["caveat"])


def test_no_disease_links_yields_no_associations() -> None:
    """An empty result, not a crash or a spurious null."""
    classes = {"P1": "c"}
    records = {"P1": FunctionalRecord(accession="P1", is_reviewed=True)}
    assert disease_associations(classes, records) == []
    assert disease_report(classes, records)["n_with_disease_association"] == 0
