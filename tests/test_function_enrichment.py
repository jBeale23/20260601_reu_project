"""Tests for functional annotation and the enrichment test built on it.

The failure mode this must avoid is rediscovering which proteins are well studied. Several
tests exist purely to pin that.
"""

from __future__ import annotations

import pytest

from data_fetching.fetch_function import FunctionalRecord, record_from_payload
from validation.function_enrichment import (
    MIN_TERM_COUNT,
    SubpopulationPair,
    annotation_coverage,
    compare_groups,
    functional_split_report,
)


def _record(
    accession: str,
    *,
    function: tuple[str, ...] = (),
    keywords: tuple[str, ...] = (),
    location: tuple[str, ...] = (),
    reviewed: bool = True,
) -> FunctionalRecord:
    return FunctionalRecord(
        accession=accession,
        go_function=tuple(function),
        keywords=tuple(keywords),
        subcellular_locations=tuple(location),
        is_reviewed=reviewed,
    )


def test_uniprot_payload_is_parsed_into_the_fields_that_matter() -> None:
    """GO aspects are split apart, because molecular function is the direct evidence."""
    payload = {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "uniProtKBCrossReferences": [
            {"database": "GO", "properties": [{"key": "GoTerm", "value": "F:Hsp70 protein binding"}]},
            {"database": "GO", "properties": [{"key": "GoTerm", "value": "P:protein folding"}]},
            {"database": "GO", "properties": [{"key": "GoTerm", "value": "C:nucleolus"}]},
            {"database": "PDB", "properties": [{"key": "Method", "value": "X-ray"}]},
        ],
        "keywords": [{"name": "Chaperone"}, {"name": "Nucleus"}],
        "comments": [
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [{"location": {"value": "Mitochondrion outer membrane"}}],
            },
            {"commentType": "INTERACTION", "interactions": [{}, {}, {}]},
        ],
    }
    record = record_from_payload("P00001", payload)

    assert record.go_function == ("Hsp70 protein binding",)
    assert record.go_process == ("protein folding",)
    assert record.go_component == ("nucleolus",)
    assert record.keywords == ("Chaperone", "Nucleus")
    assert record.subcellular_locations == ("Mitochondrion outer membrane",)
    assert record.n_interactions == 3
    assert record.is_reviewed
    assert record.n_go_terms == 3


def test_a_protein_with_no_annotation_is_marked_as_such() -> None:
    """Unannotated must be distinguishable from annotated-with-nothing-interesting."""
    empty = record_from_payload("P00002", {"entryType": "UniProtKB unreviewed (TrEMBL)"})
    assert not empty.has_any_annotation
    assert not empty.is_reviewed
    assert _record("P00003", function=("Hsp70 protein binding",)).has_any_annotation


def test_a_term_confined_to_one_group_is_detected() -> None:
    """The basic claim: a term that separates the groups should come out significant."""
    group = [_record(f"G{i}", location=("Mitochondrion outer membrane",)) for i in range(30)]
    other = [_record(f"O{i}", location=("Nucleus",)) for i in range(30)]

    results = compare_groups(group, other)
    top = results[0]
    assert top.term in {"Mitochondrion outer membrane", "Nucleus"}
    assert top.is_significant
    assert top.rate_in_group != top.rate_in_other


def test_a_term_shared_equally_is_not_flagged() -> None:
    """No difference means no finding, whatever the sample size."""
    group = [_record(f"G{i}", keywords=("Chaperone",)) for i in range(40)]
    other = [_record(f"O{i}", keywords=("Chaperone",)) for i in range(40)]

    results = compare_groups(group, other)
    assert all(not item.is_significant for item in results)


def test_rare_terms_are_not_tested() -> None:
    """A term in three proteins can reach significance while generalising to nothing.

    Excluding them also stops thousands of singletons diluting the correction.
    """
    group = [_record(f"G{i}", keywords=("Rare",) if i < 2 else ()) for i in range(40)]
    other = [_record(f"O{i}") for i in range(40)]

    tested = {item.term for item in compare_groups(group, other)}
    assert "Rare" not in tested
    assert MIN_TERM_COUNT > 3


def test_multiple_testing_correction_is_applied() -> None:
    """Thousands of terms are tested at once; raw p-values would manufacture findings."""
    group = [_record(f"G{i}", keywords=tuple(f"K{j}" for j in range(30))) for i in range(25)]
    other = [_record(f"O{i}", keywords=tuple(f"K{j}" for j in range(30))) for i in range(25)]

    results = compare_groups(group, other)
    assert results
    for item in results:
        assert item.p_adjusted >= item.p_value - 1e-12


def test_coverage_is_reported_so_study_effort_can_be_ruled_out() -> None:
    """A well-annotated group against a sparse one measures effort, not biology."""
    functions = {f"A{i}": _record(f"A{i}", function=("Hsp70 protein binding",)) for i in range(8)}
    functions.update({f"B{i}": _record(f"B{i}") for i in range(2)})

    covered = annotation_coverage([f"A{i}" for i in range(8)], functions)
    sparse = annotation_coverage([f"B{i}" for i in range(2)] + ["MISSING"], functions)

    assert covered["annotation_coverage"] == pytest.approx(1.0)
    assert sparse["annotation_coverage"] == pytest.approx(0.0)
    assert sparse["n_fetched"] == 2


def test_unannotated_proteins_are_excluded_from_the_comparison() -> None:
    """Comparing annotated against unannotated would recover study effort.

    Both groups are restricted to proteins carrying annotation before any term is counted.
    """
    functions = {}
    for i in range(20):
        functions[f"G{i}"] = _record(f"G{i}", location=("Nucleus",))
    for i in range(20):
        # Half the second group has no annotation at all.
        functions[f"O{i}"] = _record(f"O{i}", location=("Nucleus",)) if i < 10 else _record(f"O{i}")

    report = functional_split_report(
        SubpopulationPair("group", "other", [f"G{i}" for i in range(20)], [f"O{i}" for i in range(20)]),
        functions,
    )
    # Nucleus is universal among the *annotated* members of both, so nothing separates them.
    assert report["n_significant"] == 0
    assert report["coverage"]["group"]["annotation_coverage"] == pytest.approx(1.0)
    assert report["coverage"]["other"]["annotation_coverage"] == pytest.approx(0.5)


def test_empty_groups_do_not_crash() -> None:
    """A subpopulation with no annotated members yields no comparison, not an error."""
    assert compare_groups([], [_record("A")]) == []
    assert compare_groups([_record("A")], []) == []
