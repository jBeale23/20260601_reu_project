"""Tests for how proteins are chosen into an all-versus-all comparison.

The choice is a claim, not a convenience. An all-versus-all over every model is ~10^10
pairs and, staged, exhausted a shared quota and killed two unrelated jobs. A set dominated
by over-sequenced families would also answer a question about sequencing effort rather than
biology. These tests pin that the subset is chosen deliberately and reported honestly.
"""

from __future__ import annotations

import pytest

from domain_layout.records import DomainStore, ProteinDomainRecord
from validation.representatives import (
    ARCHITECTURE,
    GENUS,
    RANDOM,
    SPECIES,
    STRATEGIES,
    SelectionOptions,
    choose_representatives,
    representative_summary,
    species_of,
)


def _store() -> DomainStore:
    proteins = {
        # Two proteins sharing an architecture and a species; one is a better model.
        "P1": ProteinDomainRecord(
            accession="P1", sequence="A" * 100, ida_accession="IDA1", organism_name="Homo sapiens"
        ),
        "P2": ProteinDomainRecord(
            accession="P2", sequence="A" * 300, ida_accession="IDA1", organism_name="Homo sapiens"
        ),
        # A different architecture, different genus.
        "P3": ProteinDomainRecord(
            accession="P3", sequence="A" * 200, ida_accession="IDA2", organism_name="Saccharomyces cerevisiae"
        ),
        # Same genus as P3, different species and architecture.
        "P4": ProteinDomainRecord(
            accession="P4", sequence="A" * 150, ida_accession="IDA3", organism_name="Saccharomyces uvarum"
        ),
    }
    return DomainStore(proteins=proteins)


def test_one_representative_per_architecture() -> None:
    """The project's own unit: one example per architecture, not per sequenced genome."""
    chosen = choose_representatives(_store(), options=SelectionOptions(strategy=ARCHITECTURE))
    assert len(chosen) == 3
    # P2 beats P1 within IDA1 on length, absent structural quality scores.
    assert "P2" in chosen
    assert "P1" not in chosen


def test_structural_quality_decides_the_representative() -> None:
    """A low-confidence model must not speak for its whole architecture."""
    chosen = choose_representatives(
        _store(),
        options=SelectionOptions(strategy=ARCHITECTURE),
        quality={"P1": 95.0, "P2": 40.0},
    )
    # P1 is shorter but far better resolved, so it represents IDA1.
    assert "P1" in chosen
    assert "P2" not in chosen


def test_species_and_genus_collapse_differently() -> None:
    """Genus is coarser, so it must never yield more groups than species."""
    store = _store()
    by_species = choose_representatives(store, options=SelectionOptions(strategy=SPECIES))
    by_genus = choose_representatives(store, options=SelectionOptions(strategy=GENUS))
    assert len(by_genus) <= len(by_species)
    # P3 and P4 are both Saccharomyces, so genus merges them and species does not.
    assert len(by_species) == 3
    assert len(by_genus) == 2


def test_random_selection_is_reproducible_from_its_seed() -> None:
    """A sample that changes between runs cannot be reported."""
    store = _store()
    first = choose_representatives(store, options=SelectionOptions(strategy=RANDOM, seed=7))
    second = choose_representatives(store, options=SelectionOptions(strategy=RANDOM, seed=7))
    assert first == second


def test_a_cap_costs_size_not_quality() -> None:
    """Truncating alphabetically would silently bias the sample."""
    chosen = choose_representatives(
        _store(),
        options=SelectionOptions(strategy=ARCHITECTURE, limit=1),
        quality={"P1": 10.0, "P2": 20.0, "P3": 99.0, "P4": 50.0},
    )
    assert chosen == ["P3"]


def test_the_cap_never_ranks_plddt_against_sequence_length() -> None:
    """Mixing the two scales let a long unscored protein outrank a well-resolved one.

    Within a group the fallback is fine - a group either has structural scores or it does
    not. Across groups it is not, so the cap ranks on the structural score alone and a
    representative without one sorts last: it has no model to contribute anyway.
    """
    chosen = choose_representatives(
        _store(),
        options=SelectionOptions(strategy=ARCHITECTURE, limit=1),
        # P1/P2 (IDA1) have no structural score; their fallback length is 300.
        quality={"P3": 60.0},
    )
    assert chosen == ["P3"], "an unscored protein outranked a resolved one"


def test_a_protein_without_an_architecture_is_its_own_group() -> None:
    """Otherwise one protein would represent an enormous 'unknown' bucket."""
    store = DomainStore(
        proteins={
            "P1": ProteinDomainRecord(accession="P1", sequence="AAA", ida_accession=""),
            "P2": ProteinDomainRecord(accession="P2", sequence="AAA", ida_accession=""),
        },
    )
    assert len(choose_representatives(store, options=SelectionOptions(strategy=ARCHITECTURE))) == 2


def test_restricting_to_proteins_with_models() -> None:
    """Only proteins with an AlphaFold model can enter a structural comparison."""
    chosen = choose_representatives(
        _store(),
        options=SelectionOptions(strategy=ARCHITECTURE),
        restrict_to=["P3", "P4"],
    )
    assert set(chosen) == {"P3", "P4"}


def test_an_unknown_strategy_is_refused() -> None:
    """Silently defaulting would misreport what the sample represents."""
    with pytest.raises(ValueError, match="unknown strategy"):
        choose_representatives(_store(), options=SelectionOptions(strategy="whatever"))


def test_the_summary_reports_coverage() -> None:
    """4,155 representatives standing for 181,526 proteins is itself part of the result."""
    store = _store()
    chosen = choose_representatives(store, options=SelectionOptions(strategy=ARCHITECTURE))
    summary = representative_summary(store, chosen, strategy=ARCHITECTURE)
    assert summary["n_representatives"] == 3
    assert summary["n_proteins_in_store"] == 4
    assert summary["n_architectures_covered"] == 3


def test_species_extraction_takes_genus_and_epithet() -> None:
    """Strain suffixes must not split one species into many."""
    assert species_of("Saccharomyces cerevisiae S288C") == "Saccharomyces cerevisiae"
    assert species_of("Homo sapiens") == "Homo sapiens"


def test_every_strategy_is_reachable() -> None:
    """All four exist so a result can be checked against more than one choice."""
    assert set(STRATEGIES) == {ARCHITECTURE, SPECIES, GENUS, RANDOM}
