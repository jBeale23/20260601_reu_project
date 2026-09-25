"""Tests for superkingdom resolution."""

from __future__ import annotations

import json

from data_fetching.fetch_taxonomy import (
    ARCHAEA,
    BACTERIA,
    EUKARYOTA,
    UNKNOWN,
    TaxonomyStore,
    superkingdom_from_payload,
)


def _payload(lineage: list[str], scientific: str = "Escherichia coli") -> dict:
    """A UniProt taxonomy response carrying the given lineage."""
    return {
        "results": [
            {
                "scientificName": scientific,
                "lineage": [{"scientificName": name} for name in lineage],
            }
        ]
    }


def test_superkingdom_is_read_from_the_lineage() -> None:
    """The domain of life is whichever of the three appears in the lineage."""
    payload = _payload(["Enterobacteriaceae", "Proteobacteria", "Bacteria", "cellular organisms"])
    kingdom, lineage = superkingdom_from_payload(payload)
    assert kingdom == BACTERIA
    assert "Proteobacteria" in lineage


def test_each_domain_of_life_is_recognised() -> None:
    """All three, since a missed one would silently become 'unknown' and be dropped."""
    assert superkingdom_from_payload(_payload(["Eukaryota", "cellular organisms"]))[0] == EUKARYOTA
    assert superkingdom_from_payload(_payload(["Archaea", "cellular organisms"]))[0] == ARCHAEA
    assert superkingdom_from_payload(_payload(["Bacteria", "cellular organisms"]))[0] == BACTERIA


def test_the_top_lineage_element_is_not_taken_as_the_answer() -> None:
    """'cellular organisms' sits above every domain and would resolve everything to itself."""
    kingdom, _ = superkingdom_from_payload(_payload(["Bacteria", "cellular organisms"]))
    assert kingdom == BACTERIA


def test_an_organism_that_is_itself_a_domain_resolves() -> None:
    """A query resolving to 'Bacteria' has its answer in its own name, not its lineage."""
    payload = {"results": [{"scientificName": "Bacteria", "lineage": [{"scientificName": "cellular organisms"}]}]}
    assert superkingdom_from_payload(payload)[0] == BACTERIA


def test_an_empty_or_unrecognised_response_is_unknown_not_a_guess() -> None:
    """No result must not silently become a kingdom."""
    assert superkingdom_from_payload({"results": []})[0] == UNKNOWN
    assert superkingdom_from_payload(_payload(["some clade"]))[0] == UNKNOWN


def test_store_counts_cover_every_category() -> None:
    """Counts must include zero categories, so a missing kingdom is visible as zero."""
    store = TaxonomyStore()
    store.superkingdom["E. coli"] = BACTERIA
    store.superkingdom["Homo sapiens"] = EUKARYOTA
    counts = store.counts()
    assert counts[BACTERIA] == 1
    assert counts[EUKARYOTA] == 1
    assert counts[ARCHAEA] == 0


def test_store_round_trips_through_json() -> None:
    """Checkpoints must reload exactly."""
    store = TaxonomyStore()
    store.superkingdom["E. coli"] = BACTERIA
    store.lineage["E. coli"] = ["Bacteria"]
    restored = json.loads(json.dumps(store.to_json_dict()))
    assert restored["superkingdom"]["E. coli"] == BACTERIA
