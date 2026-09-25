"""Tests for evidence-tiered GO annotation from QuickGO."""

from __future__ import annotations

import json

import pytest

from data_fetching.fetch_go_evidence import (
    TIER_CURATED,
    TIER_ELECTRONIC,
    TIER_EXPERIMENTAL,
    TIER_PHYLOGENETIC,
    EvidenceRecord,
    EvidenceStore,
    record_from_payload,
    tier_for,
)


def _payload(rows: list[tuple[str, str]]) -> dict:
    """Build a QuickGO-shaped payload from (evidence code, term name) pairs."""
    return {"results": [{"goEvidence": code, "goName": term} for code, term in rows]}


def test_evidence_codes_map_to_their_tiers() -> None:
    """Each code lands in the tier that describes what it actually asserts."""
    assert tier_for("IDA") == TIER_EXPERIMENTAL
    assert tier_for("IBA") == TIER_PHYLOGENETIC
    assert tier_for("ISS") == TIER_CURATED
    assert tier_for("IEA") == TIER_ELECTRONIC


def test_no_data_and_unknown_codes_assert_nothing() -> None:
    """ND explicitly says nothing is known, so it must not become an annotation."""
    assert tier_for("ND") is None
    assert tier_for(None) is None
    assert tier_for("NOT_A_CODE") is None


def test_a_term_is_filed_at_its_best_evidence() -> None:
    """The same term annotated twice belongs to the stronger tier, not both.

    Filing it in both would make the electronic tier appear to contain experimental
    knowledge, which is exactly the confusion the tiers exist to prevent.
    """
    record = record_from_payload("P1", _payload([("IEA", "protein folding"), ("IDA", "protein folding")]))
    assert record.experimental == ("protein folding",)
    assert record.electronic == ()


def test_distinct_terms_land_in_distinct_tiers() -> None:
    """Different terms keep their own evidence."""
    record = record_from_payload("P1", _payload([("IDA", "a"), ("IBA", "b"), ("ISS", "c"), ("IEA", "d")]))
    assert record.experimental == ("a",)
    assert record.phylogenetic == ("b",)
    assert record.curated == ("c",)
    assert record.electronic == ("d",)


def test_rows_without_a_usable_term_are_skipped() -> None:
    """A row with no name and no id cannot become a term."""
    payload = {"results": [{"goEvidence": "IDA"}, {"goEvidence": "IDA", "goId": "GO:0001"}]}
    assert record_from_payload("P1", payload).experimental == ("GO:0001",)


def test_terms_at_or_above_is_cumulative() -> None:
    """Asking for curated includes the better-supported tiers, not only curated."""
    record = EvidenceRecord("P1", experimental=("a",), phylogenetic=("b",), curated=("c",), electronic=("d",))
    assert record.terms_at_or_above(TIER_EXPERIMENTAL) == ("a",)
    assert record.terms_at_or_above(TIER_PHYLOGENETIC) == ("a", "b")
    assert record.terms_at_or_above(TIER_CURATED) == ("a", "b", "c")
    assert record.terms_at_or_above(TIER_ELECTRONIC) == ("a", "b", "c", "d")


def test_terms_at_or_above_rejects_an_unknown_tier() -> None:
    """A typo must return nothing rather than silently falling back to everything."""
    assert EvidenceRecord("P1", experimental=("a",)).terms_at_or_above("nonsense") == ()


def test_store_counts_proteins_not_terms() -> None:
    """A tier count is how many proteins have any term there, not how many terms exist."""
    store = EvidenceStore()
    store.records["P1"] = EvidenceRecord("P1", experimental=("a", "b", "c"))
    store.records["P2"] = EvidenceRecord("P2", electronic=("d",))
    counts = store.tier_counts()
    assert counts[TIER_EXPERIMENTAL] == 1
    assert counts[TIER_ELECTRONIC] == 1


def test_terms_by_accession_omits_proteins_with_nothing_at_that_tier() -> None:
    """A protein with only electronic terms must not appear in the experimental set."""
    store = EvidenceStore()
    store.records["P1"] = EvidenceRecord("P1", experimental=("a",))
    store.records["P2"] = EvidenceRecord("P2", electronic=("d",))
    assert set(store.terms_by_accession(TIER_EXPERIMENTAL)) == {"P1"}


def test_cumulative_lookup_widens_the_set() -> None:
    """Cumulative mode picks up proteins annotated at any better-supported tier."""
    store = EvidenceStore()
    store.records["P1"] = EvidenceRecord("P1", experimental=("a",))
    store.records["P2"] = EvidenceRecord("P2", curated=("c",))
    assert set(store.terms_by_accession(TIER_CURATED, cumulative=True)) == {"P1", "P2"}
    assert set(store.terms_by_accession(TIER_CURATED)) == {"P2"}


def test_record_round_trips_through_json() -> None:
    """Checkpoints must reload exactly, or a resumed run silently loses annotation."""
    original = EvidenceRecord("P1", experimental=("a",), phylogenetic=("b",), curated=("c",), electronic=("d",))
    restored = EvidenceRecord.from_json_dict(json.loads(json.dumps(original.to_json_dict())))
    assert restored == original


def test_empty_payload_yields_an_empty_record() -> None:
    """A protein with no annotation is a record with no terms, not a failure."""
    record = record_from_payload("P1", {"results": []})
    assert record.accession == "P1"
    assert record.experimental == ()


@pytest.mark.parametrize("code", ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"])
def test_every_experimental_code_is_recognised(code: str) -> None:
    """The full experimental set, since a missed code silently shrinks the gold standard."""
    assert tier_for(code) == TIER_EXPERIMENTAL
