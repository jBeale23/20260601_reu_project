"""Extract J-domain sequence from InterPro protein records or UniProt."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from typing import Any

from jdp_classifier.constants import J_DOMAIN_INTERPRO, J_DOMAIN_PFAM

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
FASTA_HEADER_PATTERN = re.compile(r"^>.*\n", re.MULTILINE)


def get_protein_sequence(protein: dict[str, Any]) -> str | None:
    """Return full protein sequence from an InterPro protein record if present."""
    metadata = protein.get("metadata", {})
    sequence = metadata.get("sequence")
    if isinstance(sequence, str) and sequence:
        return sequence
    if isinstance(sequence, dict):
        value = sequence.get("value") or sequence.get("sequence")
        if isinstance(value, str) and value:
            return value

    top_level = protein.get("sequence")
    if isinstance(top_level, str) and top_level:
        return top_level
    if isinstance(top_level, dict):
        value = top_level.get("value")
        if isinstance(value, str) and value:
            return value
    return None


def _is_j_domain_entry(entry: dict[str, Any]) -> bool:
    accession = str(entry.get("accession", ""))
    if accession in {J_DOMAIN_INTERPRO, J_DOMAIN_PFAM}:
        return True
    if accession.startswith("PF") and J_DOMAIN_PFAM in accession:
        return True
    entry_type = entry.get("type")
    return entry_type == J_DOMAIN_PFAM


def find_j_domain_entry(protein: dict[str, Any]) -> dict[str, Any] | None:
    """Return the J-domain entry dict from protein entries, if any."""
    for entry in protein.get("entries", []):
        if _is_j_domain_entry(entry):
            return entry
    return None


def get_j_domain_coordinates(entry: dict[str, Any]) -> tuple[int, int] | None:
    """Return 1-based inclusive start/end for the J-domain from entry locations."""
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


def slice_sequence(sequence: str, start: int, end: int) -> str:
    """Slice a 1-based inclusive UniProt sequence interval."""
    return sequence[start - 1 : end]


def extract_j_domain_from_interpro(protein: dict[str, Any]) -> tuple[str | None, tuple[int, int] | None]:
    """Extract J-domain subsequence from InterPro entry coordinates."""
    sequence = get_protein_sequence(protein)
    if not sequence:
        return None, None

    entry = find_j_domain_entry(protein)
    if entry is None:
        return None, None

    coords = get_j_domain_coordinates(entry)
    if coords is None:
        return None, None

    start, end = coords
    if start < 1 or end > len(sequence) or start > end:
        return None, coords
    return slice_sequence(sequence, start, end), coords


def parse_fasta(text: str) -> str:
    """Return sequence letters from a FASTA document."""
    return FASTA_HEADER_PATTERN.sub("", text).replace("\n", "").strip()


def fetch_uniprot_fasta(
    accession: str,
    *,
    cache: dict[str, str | None] | None = None,
    allow_fetch: bool = True,
) -> str | None:
    """Fetch full UniProt FASTA for an accession, with optional in-memory cache."""
    if cache is not None and accession in cache:
        return cache[accession]

    if not allow_fetch:
        if cache is not None:
            cache[accession] = None
        return None

    url = UNIPROT_FASTA_URL.format(accession=accession)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            fasta_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        if cache is not None:
            cache[accession] = None
        return None

    sequence = parse_fasta(fasta_text)
    if cache is not None:
        cache[accession] = sequence or None
    return sequence or None


def extract_j_domain_with_fallback(
    protein: dict[str, Any],
    *,
    allow_fetch: bool = True,
    cache: dict[str, str | None] | None = None,
) -> tuple[str | None, str, tuple[int, int] | None]:
    """Return J-domain subsequence, source label, and optional coordinates."""
    subsequence, coords = extract_j_domain_from_interpro(protein)
    if subsequence:
        return subsequence, "interpro", coords

    metadata = protein.get("metadata", {})
    accession = metadata.get("accession")
    if not accession:
        return None, "missing", coords

    full_sequence = fetch_uniprot_fasta(str(accession), cache=cache, allow_fetch=allow_fetch)
    if not full_sequence:
        return None, "missing", coords

    entry = find_j_domain_entry(protein)
    if entry is not None:
        coords = get_j_domain_coordinates(entry)
        if coords is not None:
            start, end = coords
            if 1 <= start <= end <= len(full_sequence):
                return slice_sequence(full_sequence, start, end), "uniprot", coords

    return None, "missing", coords
