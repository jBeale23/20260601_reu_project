"""Tests for the domain-store schema in domain_layout/records.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from domain_layout.records import (
    DOMAIN_STORE_VERSION,
    DomainEntry,
    DomainFragment,
    DomainStore,
    ProteinDomainRecord,
    load_domain_store,
    protein_dict_from_record,
    write_domain_store,
)
from tests.conftest import make_entry, make_record

if TYPE_CHECKING:
    from pathlib import Path


def test_fragment_length_and_round_trip() -> None:
    """Fragments are 1-based inclusive and survive serialization."""
    fragment = DomainFragment(start=5, end=14)
    assert fragment.length == 10
    assert DomainFragment.from_json_dict(fragment.to_json_dict()) == fragment


def test_entry_span_uses_all_fragments() -> None:
    """A discontinuous entry spans from its first to its last fragment."""
    entry = make_entry("PF01556", 117, 143, extra_fragments=((205, 330),))
    assert entry.start == 117
    assert entry.end == 330
    assert entry.covered_length == (143 - 117 + 1) + (330 - 205 + 1)


def test_entry_without_fragments_has_zero_span() -> None:
    """Missing coordinates degrade to a zero span rather than raising."""
    entry = DomainEntry(
        accession="PF00226",
        source_database="pfam",
        entry_type="domain",
        name="DnaJ",
        integrated="",
        fragments=(),
    )
    assert entry.start == 0
    assert entry.end == 0
    assert entry.covered_length == 0


def test_record_entry_accessions_include_integrated_parents(dnaj_record: ProteinDomainRecord) -> None:
    """Entry lookups cover member-database accessions and their InterPro parents."""
    accessions = dnaj_record.entry_accessions()
    assert "PF00226" in accessions
    assert "IPR001623" in accessions
    assert dnaj_record.entries_for_database("pfam")
    assert not dnaj_record.entries_for_database("cdd")


def test_store_add_clears_previous_failure() -> None:
    """A successful record supersedes an earlier failure for the same accession."""
    store = DomainStore()
    store.add_failure("P08622", "not_found")
    assert store.failures == {"P08622": "not_found"}

    store.add(make_record("P08622", "MHPDK"))
    assert "P08622" in store
    assert store.failures == {}


def test_store_add_failure_does_not_overwrite_record() -> None:
    """Failures never displace an already-fetched record."""
    store = DomainStore()
    store.add(make_record("P08622", "MHPDK"))
    store.add_failure("P08622", "not_found")
    assert store.failures == {}


def test_store_merge_prefers_records_over_failures() -> None:
    """Merging chunk stores keeps successes and carries unresolved failures."""
    first = DomainStore()
    first.add_failure("P08622", "retries_exhausted")
    first.add_failure("Q00000", "not_found")

    second = DomainStore()
    second.add(make_record("P08622", "MHPDK"))

    first.merge(second)
    assert "P08622" in first
    assert first.failures == {"Q00000": "not_found"}
    assert len(first) == 1


def test_store_iteration_is_accession_sorted() -> None:
    """Iteration order is deterministic for reproducible outputs."""
    store = DomainStore()
    for accession in ("Q9ZZZZ", "A0A000", "P08622"):
        store.add(make_record(accession, "MHPDK"))
    assert [record.accession for record in store] == ["A0A000", "P08622", "Q9ZZZZ"]


def test_store_round_trip(tmp_path: Path, domain_store: DomainStore) -> None:
    """A written store reloads with identical records."""
    domain_store.add_failure("Q00000", "not_found")
    path = tmp_path / "store.json"
    write_domain_store(domain_store, path)

    reloaded = load_domain_store(path)
    assert len(reloaded) == len(domain_store)
    assert reloaded.failures == {"Q00000": "not_found"}
    assert reloaded.proteins["P08622"] == domain_store.proteins["P08622"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["store_version"] == DOMAIN_STORE_VERSION
    assert payload["n_with_sequence"] == 2


def test_load_domain_store_rejects_non_store(tmp_path: Path) -> None:
    """A JSON file without a proteins mapping is rejected with a clear message."""
    path = tmp_path / "not_a_store.json"
    path.write_text(json.dumps({"architectures": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like a domain store"):
        load_domain_store(path)


def test_load_domain_store_rejects_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON raises ValueError instead of a JSONDecodeError."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_domain_store(path)


def test_protein_dict_adapter_matches_interpro_shape(dnaj_record: ProteinDomainRecord) -> None:
    """The adapter emits the metadata/entries shape the older modules expect."""
    adapted = protein_dict_from_record(dnaj_record)
    assert adapted["metadata"]["accession"] == "P08622"
    assert adapted["metadata"]["sequence"].startswith("MAKQ")
    first_entry = adapted["entries"][0]
    assert first_entry["accession"] == "PF00226"
    assert first_entry["entry_protein_locations"][0]["fragments"][0] == {"start": 5, "end": 67}
