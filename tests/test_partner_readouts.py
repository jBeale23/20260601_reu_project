"""Tests for the shared partner readouts and the partition relationship diagnostic."""

from __future__ import annotations

import pytest

from validation.partner_readouts import (
    COARSENING,
    CROSSING,
    REFINEMENT,
    RELABELING,
    describe_relationship,
    dominant_label,
    interactome_scope,
    partition_relationship,
    tie_rate,
)


def test_dominant_label_picks_the_commonest_partner_class() -> None:
    """The readout is the modal label, not the first or the rarest."""
    interactions = {"P1": ["a", "a", "b"]}
    assert dominant_label(interactions, lambda g: g.upper()) == {"P1": "A"}


def test_dominant_label_drops_proteins_with_no_usable_partner() -> None:
    """A protein whose partners all map to nothing is absent, not bucketed as unassigned.

    An "unassigned" bucket would grow large enough to dominate the contingency table and
    would then be measuring annotation coverage rather than biology.
    """
    interactions = {"P1": ["x"], "P2": ["a"]}
    result = dominant_label(interactions, lambda g: None if g == "x" else "SYS")
    assert result == {"P2": "SYS"}


def test_dominant_label_treats_unassigned_as_no_label() -> None:
    """The literal string 'unassigned' is discarded for the same reason as None."""
    interactions = {"P1": ["a", "b"]}
    assert dominant_label(interactions, lambda g: "unassigned" if g == "a" else "SYS") == {"P1": "SYS"}


def test_interactome_scope_splits_on_the_threshold() -> None:
    """One Hsp70 in a hundred partners clears the 1% cut; one in two hundred does not."""
    hsp70 = {"HSPA1A"}
    rich = {"P1": ["HSPA1A", *[f"g{i}" for i in range(99)]]}
    poor = {"P2": ["HSPA1A", *[f"g{i}" for i in range(199)]]}
    assert interactome_scope(rich, lambda g: g in hsp70)["P1"] == "hsp70_rich"
    assert interactome_scope(poor, lambda g: g in hsp70)["P2"] == "hsp70_poor"


def test_interactome_scope_skips_empty_interactomes() -> None:
    """A protein with no partners has no scope, and must not divide by zero."""
    assert interactome_scope({"P1": []}, lambda _g: True) == {}


def test_relabeling_is_recognised() -> None:
    """Renaming A to X and B to Y is the same partition and can never be a finding."""
    reference = {"P1": "A", "P2": "A", "P3": "B"}
    candidate = {"P1": "X", "P2": "X", "P3": "Y"}
    assert partition_relationship(candidate, reference) == RELABELING


def test_refinement_is_recognised() -> None:
    """Splitting A into two groups while leaving B alone is a refinement."""
    reference = {"P1": "A", "P2": "A", "P3": "B"}
    candidate = {"P1": "A1", "P2": "A2", "P3": "B"}
    assert partition_relationship(candidate, reference) == REFINEMENT


def test_coarsening_is_recognised() -> None:
    """Merging B and C against A is a coarsening - the real shape of has_zinc_finger_like."""
    reference = {"P1": "A", "P2": "B", "P3": "C"}
    candidate = {"P1": "yes", "P2": "no", "P3": "no"}
    assert partition_relationship(candidate, reference) == COARSENING


def test_crossing_partition_is_recognised() -> None:
    """A partition that cuts across the reference in both directions is genuinely new."""
    reference = {"P1": "A", "P2": "A", "P3": "B", "P4": "B"}
    candidate = {"P1": "X", "P2": "Y", "P3": "X", "P4": "Y"}
    assert partition_relationship(candidate, reference) == CROSSING


def test_relationship_with_no_shared_proteins_is_crossing() -> None:
    """Nothing in common means nothing can be concluded about nesting."""
    assert partition_relationship({"P1": "X"}, {"P2": "A"}) == CROSSING


def test_every_relationship_has_a_description() -> None:
    """The report prints these verbatim, so none may be missing."""
    for relationship in (RELABELING, REFINEMENT, COARSENING, CROSSING):
        assert len(describe_relationship(relationship)) > 20


def test_ties_are_dropped_by_default() -> None:
    """A protein whose partners split evenly has no dominant partner, so it is excluded.

    Assigning one would invent an observation, and the arbitrary pick is exactly what made
    the original readout non-reproducible between runs.
    """
    interactions = {"P1": ["a", "b"], "P2": ["a", "a", "b"]}
    result = dominant_label(interactions, lambda g: g.upper())
    assert result == {"P2": "A"}


def test_first_on_tie_is_deterministic() -> None:
    """The opt-in tie-break must give the same answer every run, unlike set iteration."""
    interactions = {"P1": ["b", "a"]}
    first = dominant_label(interactions, lambda g: g.upper(), on_tie="first")
    second = dominant_label(interactions, lambda g: g.upper(), on_tie="first")
    assert first == second == {"P1": "A"}


def test_unknown_tie_policy_is_rejected() -> None:
    """A typo in the policy must fail loudly rather than silently dropping a quarter of the data."""
    with pytest.raises(ValueError, match="on_tie"):
        dominant_label({"P1": ["a"]}, lambda g: g.upper(), on_tie="whatever")


def test_tie_rate_reports_how_much_is_ambiguous() -> None:
    """The fraction of tied proteins is part of the result, not a hidden detail."""
    interactions = {"P1": ["a", "b"], "P2": ["a", "a"], "P3": ["c"], "P4": []}
    rate = tie_rate(interactions, lambda g: g.upper())
    assert rate["n_labelled"] == 3
    assert rate["n_tied"] == 1
    assert rate["tie_fraction"] == pytest.approx(1 / 3, abs=0.001)


def test_tie_rate_handles_no_labelled_proteins() -> None:
    """No labels means no tie rate, not a division by zero."""
    assert tie_rate({"P1": ["x"]}, lambda _g: None)["tie_fraction"] == 0.0
