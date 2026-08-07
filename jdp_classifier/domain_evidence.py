"""Use fetched InterPro domain records as classifier input.

The DnaJ architecture fetch returns protein metadata only — no sequence and no domain
coordinates — so HPD detection and localization fall back to per-accession UniProt
requests. When a domain store from ``fetch-protein-domains`` is available, this module
injects the fetched sequence, entries, and Pfam architecture into each protein record so
the classifier runs offline and on complete evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domain_layout.records import protein_dict_from_record

if TYPE_CHECKING:
    from domain_layout.records import DomainStore, ProteinDomainRecord

PFAM_DATABASE = "pfam"


def ida_from_record(record: ProteinDomainRecord) -> str:
    """Rebuild an InterPro IDA string from a record's Pfam matches, ordered by position.

    Produces the same ``PF00226:IPR001623-PF01556:IPR002939`` shape as the architecture
    fetch, so downstream architecture rules work unchanged.
    """
    segments: list[tuple[int, str]] = []
    seen: set[str] = set()
    for entry in record.entries:
        if entry.source_database != PFAM_DATABASE or entry.accession in seen:
            continue
        seen.add(entry.accession)
        label = f"{entry.accession}:{entry.integrated}" if entry.integrated else entry.accession
        segments.append((entry.start, label))
    return "-".join(label for _start, label in sorted(segments))


def enrich_protein(protein: dict[str, Any], record: ProteinDomainRecord | None) -> dict[str, Any]:
    """Return the protein record with fetched sequence and entries merged in.

    Fetch metadata (name, length, source database) wins over store metadata so merged
    tables stay consistent with the fetch JSON; the sequence and entries come from the
    store because the fetch does not provide them.
    """
    if record is None:
        return protein

    adapted = protein_dict_from_record(record)
    metadata = {**adapted["metadata"], **{key: value for key, value in protein.get("metadata", {}).items() if value}}
    metadata["sequence"] = record.sequence

    enriched = dict(protein)
    enriched["metadata"] = metadata
    enriched["entries"] = adapted["entries"] or protein.get("entries", [])
    return enriched


def record_for_accession(store: DomainStore | None, accession: str) -> ProteinDomainRecord | None:
    """Look up one accession in an optional domain store."""
    if store is None or not accession:
        return None
    return store.proteins.get(accession)
