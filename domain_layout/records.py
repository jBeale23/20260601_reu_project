"""Protein domain records: the shared on-disk schema for fetched InterPro domains.

A *domain store* is the JSON file written by ``fetch-protein-domains``. It holds one
:class:`ProteinDomainRecord` per UniProt accession: the full sequence plus every
InterPro and member-database signature match with residue coordinates. Every
downstream layer (region routing, disorder prediction, SHARK scoring, the JDP
classifier, motif conservation) reads this schema instead of re-querying the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DOMAIN_STORE_VERSION = 1

# InterPro member databases whose matches are kept as domain evidence, ordered by
# how well their boundaries correspond to a single structural unit.
MEMBER_DATABASE_PRIORITY = (
    "pfam",
    "interpro",
    "cdd",
    "profile",
    "smart",
    "cathgene3d",
    "ssf",
    "ncbifam",
    "hamap",
    "panther",
    "prints",
    "prosite",
)


@dataclass(frozen=True, slots=True, order=True)
class DomainFragment:
    """One contiguous residue interval of a domain match (1-based, inclusive)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of residues covered by this fragment."""
        return max(0, self.end - self.start + 1)

    def to_json_dict(self) -> dict[str, int]:
        """Serialize to a JSON-native dict."""
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DomainFragment:
        """Build a fragment from a JSON-native dict."""
        return cls(start=int(payload["start"]), end=int(payload["end"]))


@dataclass(frozen=True, slots=True)
class DomainEntry:
    """One InterPro or member-database signature match on a protein."""

    accession: str
    source_database: str
    entry_type: str
    name: str
    integrated: str
    fragments: tuple[DomainFragment, ...]

    @property
    def start(self) -> int:
        """First residue covered by any fragment (0 when the entry has no fragments)."""
        return min((fragment.start for fragment in self.fragments), default=0)

    @property
    def end(self) -> int:
        """Last residue covered by any fragment (0 when the entry has no fragments)."""
        return max((fragment.end for fragment in self.fragments), default=0)

    @property
    def covered_length(self) -> int:
        """Total residues covered by the entry's fragments."""
        return sum(fragment.length for fragment in self.fragments)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-native dict."""
        return {
            "accession": self.accession,
            "source_database": self.source_database,
            "entry_type": self.entry_type,
            "name": self.name,
            "integrated": self.integrated,
            "fragments": [fragment.to_json_dict() for fragment in self.fragments],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DomainEntry:
        """Build an entry from a JSON-native dict."""
        fragments = tuple(DomainFragment.from_json_dict(item) for item in payload.get("fragments", []))
        return cls(
            accession=str(payload.get("accession", "")),
            source_database=str(payload.get("source_database", "")),
            entry_type=str(payload.get("entry_type", "")),
            name=str(payload.get("name", "")),
            integrated=str(payload.get("integrated") or ""),
            fragments=tuple(sorted(fragments)),
        )


@dataclass(frozen=True, slots=True)
class ProteinDomainRecord:
    """Sequence plus complete domain annotation for one UniProt accession."""

    accession: str
    name: str = ""
    length: int = 0
    sequence: str = ""
    source_database: str = ""
    organism_tax_id: str = ""
    organism_name: str = ""
    ida_accession: str = ""
    is_fragment: bool = False
    entries: tuple[DomainEntry, ...] = ()

    @property
    def is_probable_fragment(self) -> bool:
        """Whether UniProt flags this entry as a sequence fragment.

        Fragments matter for classification: an incomplete gene model looks exactly like
        "a J-domain with no partner domains", which is the dominant novelty signal.
        """
        return self.is_fragment

    @property
    def has_sequence(self) -> bool:
        """Whether a usable amino-acid sequence was fetched."""
        return bool(self.sequence)

    def entries_for_database(self, source_database: str) -> tuple[DomainEntry, ...]:
        """Return entries from one member database (e.g. ``pfam``)."""
        return tuple(entry for entry in self.entries if entry.source_database == source_database)

    def entry_accessions(self) -> frozenset[str]:
        """Return every signature accession on this protein, plus integrated InterPro parents."""
        accessions = {entry.accession for entry in self.entries}
        accessions |= {entry.integrated for entry in self.entries if entry.integrated}
        return frozenset(accessions)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-native dict."""
        return {
            "accession": self.accession,
            "name": self.name,
            "length": self.length,
            "sequence": self.sequence,
            "source_database": self.source_database,
            "organism_tax_id": self.organism_tax_id,
            "organism_name": self.organism_name,
            "ida_accession": self.ida_accession,
            "is_fragment": self.is_fragment,
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProteinDomainRecord:
        """Build a record from a JSON-native dict."""
        entries = tuple(DomainEntry.from_json_dict(item) for item in payload.get("entries", []))
        return cls(
            accession=str(payload.get("accession", "")),
            name=str(payload.get("name", "")),
            length=int(payload.get("length") or 0),
            sequence=str(payload.get("sequence", "")),
            source_database=str(payload.get("source_database", "")),
            organism_tax_id=str(payload.get("organism_tax_id", "")),
            organism_name=str(payload.get("organism_name", "")),
            ida_accession=str(payload.get("ida_accession", "")),
            is_fragment=bool(payload.get("is_fragment", False)),
            entries=entries,
        )


@dataclass(slots=True)
class DomainStore:
    """All fetched domain records plus per-accession failure reasons."""

    proteins: dict[str, ProteinDomainRecord] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    source: str = ""
    store_version: int = DOMAIN_STORE_VERSION

    def __len__(self) -> int:
        """Number of successfully fetched proteins."""
        return len(self.proteins)

    def __contains__(self, accession: str) -> bool:
        """Whether an accession has a fetched record."""
        return accession in self.proteins

    def __iter__(self) -> Iterator[ProteinDomainRecord]:
        """Iterate records in accession order."""
        for accession in sorted(self.proteins):
            yield self.proteins[accession]

    def add(self, record: ProteinDomainRecord) -> None:
        """Insert or replace one record and clear any earlier failure for it."""
        self.proteins[record.accession] = record
        self.failures.pop(record.accession, None)

    def add_failure(self, accession: str, reason: str) -> None:
        """Record why an accession could not be fetched."""
        if accession not in self.proteins:
            self.failures[accession] = reason

    def merge(self, other: DomainStore) -> None:
        """Merge another store into this one; successful records win over failures."""
        for record in other:
            self.add(record)
        for accession, reason in other.failures.items():
            self.add_failure(accession, reason)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the whole store to a JSON-native dict."""
        return {
            "store_version": self.store_version,
            "source": self.source,
            "n_proteins": len(self.proteins),
            "n_failed": len(self.failures),
            "n_with_sequence": sum(1 for record in self if record.has_sequence),
            "proteins": {accession: self.proteins[accession].to_json_dict() for accession in sorted(self.proteins)},
            "failures": {accession: self.failures[accession] for accession in sorted(self.failures)},
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DomainStore:
        """Build a store from a JSON-native dict."""
        proteins = {
            str(accession): ProteinDomainRecord.from_json_dict(record)
            for accession, record in (payload.get("proteins") or {}).items()
        }
        failures = {str(accession): str(reason) for accession, reason in (payload.get("failures") or {}).items()}
        return cls(
            proteins=proteins,
            failures=failures,
            source=str(payload.get("source", "")),
            store_version=int(payload.get("store_version") or DOMAIN_STORE_VERSION),
        )


def write_domain_store(store: DomainStore, output_path: Path) -> None:
    """Write a domain store to disk as indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def load_domain_store(path: Path) -> DomainStore:
    """Load a domain store written by ``fetch-protein-domains``.

    Raises:
        ValueError: If the file is not valid JSON or lacks a ``proteins`` mapping.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in domain store {path}: {exc.msg}"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict) or "proteins" not in payload:
        msg = f"File does not look like a domain store (missing 'proteins'): {path}"
        raise ValueError(msg)

    return DomainStore.from_json_dict(payload)


def protein_dict_from_record(record: ProteinDomainRecord) -> dict[str, Any]:
    """Render a record in the InterPro protein-payload shape used by existing modules.

    ``jdp_classifier`` and ``motif_conservation`` consume raw InterPro protein
    dictionaries (``metadata`` + ``entries`` with ``entry_protein_locations``).
    This adapter lets them run on fetched domain records without network access.
    """
    return {
        "metadata": {
            "accession": record.accession,
            "name": record.name,
            "length": record.length,
            "source_database": record.source_database,
            "sequence": record.sequence,
            "source_organism": {
                "taxId": record.organism_tax_id,
                "fullName": record.organism_name,
            },
        },
        "entries": [
            {
                "accession": entry.accession,
                "source_database": entry.source_database,
                "type": entry.entry_type,
                "name": entry.name,
                "integrated": entry.integrated,
                "entry_protein_locations": [
                    {"fragments": [fragment.to_json_dict() for fragment in entry.fragments]},
                ],
            }
            for entry in record.entries
        ],
    }
