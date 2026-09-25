"""Extract and group DnaJ domain family sequences from InterPro fetch JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from domain_layout.records import protein_dict_from_record
from jdp_classifier.sequence import get_protein_sequence, slice_sequence
from motif_conservation.constants import (
    DOMAIN_FAMILY_LABELS,
    IDR_LIKE_DOMAIN_PFAMS,
    STRUCTURED_DOMAIN_PFAMS,
)
from scripts.extract_uniprot_ids import normalize_accession

if TYPE_CHECKING:
    from collections.abc import Container
    from pathlib import Path

    from domain_layout.records import DomainStore


@dataclass(frozen=True, slots=True)
class DomainSlice:
    """One domain instance sliced from a protein sequence."""

    accession: str
    domain_family: str
    pfam: str
    start: int
    end: int
    sequence: str


def _entry_pfam(entry: dict[str, Any]) -> str | None:
    accession = str(entry.get("accession", ""))
    if accession.startswith("PF"):
        return accession.split("-", maxsplit=1)[0]
    metadata = entry.get("metadata") or {}
    member = metadata.get("accession") or metadata.get("name")
    if isinstance(member, str) and member.startswith("PF"):
        return member.split("-", maxsplit=1)[0]
    return None


def _entry_coordinates(entry: dict[str, Any]) -> tuple[int, int] | None:
    locations = entry.get("entry_protein_locations") or entry.get("protein_locations") or []
    for location in locations:
        fragments = location.get("fragments", [])
        if not fragments:
            continue
        fragment = fragments[0]
        start = fragment.get("start")
        end = fragment.get("end")
        if start is not None and end is not None:
            return int(start), int(end)
    return None


def family_label_for_pfam(pfam: str) -> str | None:
    """Return canonical family label for a Pfam accession."""
    return DOMAIN_FAMILY_LABELS.get(pfam)


def iter_domain_slices(protein: dict[str, Any]) -> list[DomainSlice]:
    """Extract structured and IDR-like domain slices from one InterPro protein."""
    accession = normalize_accession((protein.get("metadata") or {}).get("accession"))
    sequence = get_protein_sequence(protein)
    if not accession or not sequence:
        return []

    slices: list[DomainSlice] = []
    seen: set[tuple[str, int, int]] = set()
    for entry in protein.get("entries", []):
        pfam = _entry_pfam(entry)
        if pfam is None:
            continue
        if pfam not in STRUCTURED_DOMAIN_PFAMS and pfam not in IDR_LIKE_DOMAIN_PFAMS:
            continue
        coords = _entry_coordinates(entry)
        if coords is None:
            continue
        start, end = coords
        if end < start or end > len(sequence):
            continue
        key = (pfam, start, end)
        if key in seen:
            continue
        seen.add(key)
        label = family_label_for_pfam(pfam)
        if label is None:
            continue
        slices.append(
            DomainSlice(
                accession=accession,
                domain_family=label,
                pfam=pfam,
                start=start,
                end=end,
                sequence=slice_sequence(sequence, start, end),
            ),
        )
    return slices


def group_domain_families(proteins: list[dict[str, Any]]) -> dict[str, list[DomainSlice]]:
    """Group domain slices by canonical family label."""
    grouped: dict[str, list[DomainSlice]] = defaultdict(list)
    for protein in proteins:
        for domain in iter_domain_slices(protein):
            grouped[domain.domain_family].append(domain)
    return dict(grouped)


def load_dnaj_proteins(fetch_json: Path) -> list[dict[str, Any]]:
    """Load protein records from a DnaJ architecture fetch JSON.

    Raises:
        ValueError: If the JSON is neither an architecture nor a protein fetch file.
    """
    data = json.loads(fetch_json.read_text(encoding="utf-8"))
    proteins: list[dict[str, Any]] = []
    if "architectures" in data:
        for architecture in data["architectures"]:
            proteins.extend(architecture.get("proteins") or [])
    elif "proteins" in data:
        proteins.extend(data["proteins"])
    else:
        msg = f"Unrecognized fetch JSON layout in {fetch_json}"
        raise ValueError(msg)
    return proteins


def proteins_from_domain_store(
    store: DomainStore,
    *,
    restrict_to: Container[str] | None = None,
) -> list[dict[str, Any]]:
    """Render domain-store records as InterPro-shaped protein dicts.

    The architecture fetch carries no sequences or per-protein domain coordinates, so
    slicing domain families from it yields nothing. Records fetched by
    ``fetch-protein-domains`` carry both, which is what this adapter exposes.

    Args:
        store: Domain store loaded from ``fetch-protein-domains`` output.
        restrict_to: Optional accession set to keep (e.g. accessions from a fetch JSON).
    """
    return [
        protein_dict_from_record(record)
        for record in store
        if record.has_sequence and (restrict_to is None or record.accession in restrict_to)
    ]


def dedupe_slices_by_accession(slices: list[DomainSlice]) -> list[DomainSlice]:
    """Keep the first domain slice per accession within a family."""
    ordered: list[DomainSlice] = []
    seen: set[str] = set()
    for domain in slices:
        if domain.accession in seen:
            continue
        seen.add(domain.accession)
        ordered.append(domain)
    return ordered
