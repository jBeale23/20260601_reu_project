"""Tests for the non-Hsp70 interactome analysis.

The pairing sweep reduces every non-Hsp70 partner to a single ``non_hsp70`` bit. Of 8,638
distinct BioGRID partner genes here, 8,590 fall in that bucket, and they are not
interchangeable - ribosomal subunits and proteasome lids say different things about what a
J-domain protein does. These tests pin the finer split.
"""

from __future__ import annotations

import pytest

from validation.partner_systems import (
    MIN_PROTEINS_PER_SYSTEM,
    PARTNER_SYSTEMS,
    SIGNIFICANCE_LEVEL,
    UNASSIGNED,
    classify_non_hsp70,
    non_hsp70_report,
    system_associations,
    system_fractions,
    systems_for,
)


def test_an_hsp70_is_not_this_modules_business() -> None:
    """Hsp70 partners belong to the pairing sweep; returning a system would double-count."""
    assert classify_non_hsp70("SSA1") is None
    assert classify_non_hsp70("HSPA8") is None
    assert classify_non_hsp70("KAR2") is None


def test_known_systems_are_recognised() -> None:
    """The systems chaperone biology actually distinguishes."""
    assert classify_non_hsp70("RPL3") == "ribosome"
    assert classify_non_hsp70("PSMA1") == "proteasome"
    assert classify_non_hsp70("CCT2") == "chaperonin"
    assert classify_non_hsp70("HSP82") == "hsp90"
    assert classify_non_hsp70("ACT1") == "cytoskeleton"
    assert classify_non_hsp70("SEC61") == "protein_translocation"


def test_a_nucleotide_exchange_factor_is_not_an_hsp70() -> None:
    """NEFs act *on* Hsp70 rather than being one, so they are their own system.

    Folding them into the Hsp70 axes would make a protein that recruits an exchange factor
    look like one that partners a chaperone.
    """
    assert classify_non_hsp70("BAG1") == "nucleotide_exchange_factor"
    assert classify_non_hsp70("FES1") == "nucleotide_exchange_factor"


def test_an_unknown_symbol_is_counted_not_forced_into_a_bucket() -> None:
    """A split that only appears after discarding most partners is not a split."""
    assert classify_non_hsp70("ZZZ999") == UNASSIGNED


def test_systems_for_excludes_hsp70s_and_unassigned() -> None:
    """Only real, identified systems count toward what a protein touches."""
    assert systems_for(["SSA1", "RPL3", "ZZZ999", "PSMA1"]) == {"ribosome", "proteasome"}


def test_a_class_touching_one_system_exclusively_is_detected() -> None:
    """The basic signal the analysis exists to find."""
    classes = {f"R{index}": "ribosome_class" for index in range(10)}
    classes.update({f"P{index}": "proteasome_class" for index in range(10)})
    interactions = {f"R{index}": ["RPL3", "RPS6"] for index in range(10)}
    interactions.update({f"P{index}": ["PSMA1", "PSMB2"] for index in range(10)})

    results = system_associations(classes, interactions)
    ribosome = [item for item in results if item.system == "ribosome" and item.protein_class == "ribosome_class"]
    assert ribosome
    # All of the ribosome class's assigned partners are ribosomal; none of the other's are.
    assert ribosome[0].median_in_class == pytest.approx(1.0)
    assert ribosome[0].median_in_others == pytest.approx(0.0)
    assert ribosome[0].direction == "enriched"
    assert ribosome[0].p_adjusted < SIGNIFICANCE_LEVEL


def test_no_association_when_every_class_touches_everything() -> None:
    """A system every class touches equally carries no class information."""
    classes = {f"P{index}": "a" if index % 2 else "b" for index in range(20)}
    interactions = {f"P{index}": ["RPL3", "PSMA1"] for index in range(20)}
    results = system_associations(classes, interactions)
    assert all(item.p_adjusted > SIGNIFICANCE_LEVEL for item in results)


def test_rare_systems_are_not_quoted() -> None:
    """One protein touching a system is not a rate."""
    classes = {f"P{index}": "a" for index in range(10)}
    interactions = {f"P{index}": (["RPL3"] if index == 0 else ["ACT1"]) for index in range(10)}
    results = system_associations(classes, interactions)
    assert all(item.system != "ribosome" for item in results), "a single carrier was quoted as a rate"
    assert MIN_PROTEINS_PER_SYSTEM >= 5


def test_the_report_surfaces_how_much_went_unassigned() -> None:
    """Coverage is part of the result, not a footnote."""
    classes = {f"P{index}": "a" for index in range(10)}
    interactions = {f"P{index}": ["RPL3", "ZZZ999", "QQQ111"] for index in range(10)}
    report = non_hsp70_report(classes, interactions)
    assert report["unassigned_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert "ZZZ999" in report["most_common_unassigned_symbols"]


def test_no_shared_proteins_yields_an_empty_result() -> None:
    """Classes and interactions that share nothing cannot be tested."""
    assert system_associations({"A": "x"}, {"B": ["RPL3"]}) == []


def test_every_system_has_at_least_one_prefix() -> None:
    """An empty prefix tuple would silently match nothing."""
    assert all(prefixes for prefixes in PARTNER_SYSTEMS.values())


def test_composition_is_invariant_to_interactome_size() -> None:
    """The confound that makes the obvious version of this analysis unusable.

    Median interactome size runs from 46 partners to 679 across classes, a fifteen-fold
    spread, so "does this class touch system X" measures how well studied a protein is.
    A protein with ten ribosomal partners and one with a thousand have the same
    composition, and the measure must say so.
    """
    small = system_fractions(["RPL3", "PSMA1"])
    large = system_fractions(["RPL3"] * 500 + ["PSMA1"] * 500)
    assert small == pytest.approx(large)
    assert small["ribosome"] == pytest.approx(0.5)


def test_fractions_are_taken_over_assigned_partners_only() -> None:
    """Assigned partners are the denominator, not all partners.

    93.7% of real partner mentions match no system in the prefix table; dividing by
    all of them would measure that coverage gap, not composition.
    """
    fractions = system_fractions(["RPL3", "ZZZ999", "QQQ111", "SSA1"])
    # One assigned partner, so it is the whole of the assigned composition.
    assert fractions == {"ribosome": pytest.approx(1.0)}


def test_a_protein_with_no_assigned_partner_has_no_composition() -> None:
    """It must not enter every comparison as a run of zeros."""
    assert system_fractions(["ZZZ999", "SSA1"]) == {}


def test_the_report_surfaces_interactome_size_per_class() -> None:
    """The confound has to be visible in the output, not just handled internally."""
    classes = {"P1": "big", "P2": "small"}
    interactions = {"P1": ["RPL3"] * 100, "P2": ["PSMA1"]}
    report = non_hsp70_report(classes, interactions)
    sizes = report["interactome_size_by_class"]
    assert sizes["big"]["median_partners"] == 100
    assert sizes["small"]["median_partners"] == 1
