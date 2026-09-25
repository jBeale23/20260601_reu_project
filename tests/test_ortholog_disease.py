"""Tests for carrying disease association across orthologs.

Transfer multiplies a ten-protein labelled set into something testable, at the cost of an
assumption. These tests pin where that assumption is allowed to apply and where it is not.
"""

from __future__ import annotations

from validation.ortholog_disease import (
    DEFAULT_TRANSFER_IDENTITY,
    MIN_REGION_LENGTH,
    transfer_disease,
    transfer_report,
)

_CTD = "MSSGNQEEHKAFAKDLLTLLNSKEEQMNKFTQSFDQFMKSHFGNLDPKQFEQLQKMLSSHFDLLTQ"


def test_a_close_ortholog_receives_the_association() -> None:
    """The case transfer exists for: a mouse relative of a human disease gene."""
    donors = {"HUMAN1": ["Muscular dystrophy, limb-girdle 1"]}
    result = transfer_disease(
        donors,
        {"HUMAN1": _CTD},
        {"MOUSE1": _CTD[:-3] + "AAA"},
        organisms={"HUMAN1": "Homo sapiens", "MOUSE1": "Mus musculus"},
    )
    assert "MOUSE1" in result
    assert result["MOUSE1"].donor == "HUMAN1"
    assert result["MOUSE1"].diseases == ("Muscular dystrophy, limb-girdle 1",)
    assert result["MOUSE1"].identity > DEFAULT_TRANSFER_IDENTITY


def test_a_distant_relative_does_not() -> None:
    """The assumption breaks exactly where this project looks: the non-J regions.

    Two orthologs can share a J-domain at 90% identity and share almost nothing else, so a
    distant C-terminus means a functionally different protein.
    """
    donors = {"HUMAN1": ["Some disease"]}
    result = transfer_disease(donors, {"HUMAN1": _CTD}, {"FAR1": "W" * len(_CTD)})
    assert result == {}


def test_an_already_annotated_protein_is_not_overwritten() -> None:
    """Observed evidence outranks transferred evidence, always."""
    donors = {"HUMAN1": ["A"], "HUMAN2": ["B"]}
    result = transfer_disease(donors, {"HUMAN1": _CTD, "HUMAN2": _CTD}, {"HUMAN2": _CTD})
    assert "HUMAN2" not in result


def test_the_closest_donor_wins() -> None:
    """A protein between two annotated relatives takes the nearer one's label."""
    near = _CTD[:-2] + "AA"
    far = _CTD[:30] + "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    donors = {"NEAR": ["near disease"], "FAR": ["far disease"]}
    result = transfer_disease(
        donors,
        {"NEAR": near, "FAR": far},
        {"TARGET": _CTD},
        identity_threshold=0.3,
    )
    assert result["TARGET"].donor == "NEAR"


def test_short_regions_are_not_transferred_across() -> None:
    """Below a few dozen residues, identity is dominated by chance."""
    donors = {"D1": ["disease"]}
    assert transfer_disease(donors, {"D1": "MVKET"}, {"T1": "MVKET"}) == {}
    assert MIN_REGION_LENGTH >= 30


def test_a_transfer_is_always_marked_as_transferred() -> None:
    """A result that appears only after transfer is a result about the transfer."""
    donors = {"HUMAN1": ["Some disease"]}
    result = transfer_disease(donors, {"HUMAN1": _CTD}, {"MOUSE1": _CTD})
    assert result["MOUSE1"].to_json_dict()["evidence"] == "transferred"


def test_the_report_keeps_observed_and_transferred_apart() -> None:
    """Merging them silently would let a transfer masquerade as evidence."""
    donors = {"HUMAN1": ["Some disease"]}
    transferred = transfer_disease(
        donors,
        {"HUMAN1": _CTD},
        {"MOUSE1": _CTD},
        organisms={"MOUSE1": "Mus musculus"},
    )
    report = transfer_report(donors, transferred)
    assert report["n_observed"] == 1
    assert report["n_transferred"] == 1
    assert report["n_species_reached"] == 1
    assert "non-J regions" in str(report["note"])


def test_no_usable_donor_yields_nothing() -> None:
    """A donor whose non-J region is too short cannot license a transfer."""
    assert transfer_disease({"D1": ["x"]}, {"D1": "MV"}, {"T1": _CTD}) == {}
