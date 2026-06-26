"""Extract unique UniProt accessions from InterPro fetch JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FetchKind = Literal["dnak", "dnaj", "unknown"]


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    """Counts from an InterPro fetch JSON accession extraction."""

    fetch_kind: FetchKind
    raw_records: int
    records_with_accession: int
    unique_accessions: int
    duplicate_records_skipped: int
    proteins_reported: int | None
    total_proteins_fetched: int | None
    architecture_count: int | None
    total_architecture_instances: int | None


def normalize_accession(value: object) -> str | None:
    """Return a stripped accession string or None when missing/blank."""
    if value is None:
        return None
    accession = str(value).strip()
    return accession or None


def extract_accessions(data: dict[str, Any]) -> list[str]:
    """Return sorted unique UniProt accessions from a DnaK or DnaJ output file.

    Args:
        data: Parsed JSON from ``fetch-proteins-dnak`` or ``fetch-architectures-dnaj``.

    Returns:
        Sorted list of unique accession strings.
    """
    seen: set[str] = set()

    for protein in data.get("proteins", []):
        accession = normalize_accession(protein.get("metadata", {}).get("accession"))
        if accession:
            seen.add(accession)

    for architecture in data.get("architectures", []):
        for protein in architecture.get("proteins", []):
            accession = normalize_accession(protein.get("metadata", {}).get("accession"))
            if accession:
                seen.add(accession)

    return sorted(seen)


def fetch_warnings(data: dict[str, Any]) -> list[str]:
    """Return human-readable warnings for incomplete InterPro fetch output."""
    warnings: list[str] = []

    if data.get("is_partial"):
        warnings.append(
            "DnaK fetch output is marked is_partial=true; accession list may be incomplete.",
        )

    partial_archs = [arch for arch in data.get("architectures", []) if arch.get("is_partial")]
    if partial_archs:
        warnings.append(
            f"{len(partial_archs)} architecture(s) are partial; accession list may be incomplete.",
        )

    return warnings


def _fetch_kind(data: dict[str, Any]) -> FetchKind:
    if data.get("architectures") is not None:
        return "dnaj"
    if data.get("proteins") is not None:
        return "dnak"
    return "unknown"


def extraction_stats(data: dict[str, Any]) -> ExtractionStats:
    """Compute raw vs unique accession counts and fetch metadata for validation."""
    fetch_kind = _fetch_kind(data)
    raw_records = 0
    records_with_accession = 0
    seen: set[str] = set()

    for protein in data.get("proteins", []):
        raw_records += 1
        accession = normalize_accession(protein.get("metadata", {}).get("accession"))
        if accession:
            records_with_accession += 1
            seen.add(accession)

    total_architecture_instances = 0
    for architecture in data.get("architectures", []):
        proteins = architecture.get("proteins", [])
        total_architecture_instances += len(proteins)
        raw_records += len(proteins)
        for protein in proteins:
            accession = normalize_accession(protein.get("metadata", {}).get("accession"))
            if accession:
                records_with_accession += 1
                seen.add(accession)

    proteins_reported = data.get("proteins_reported")
    total_proteins_fetched = data.get("total_proteins_fetched")
    if proteins_reported is not None:
        proteins_reported = int(proteins_reported)
    if total_proteins_fetched is not None:
        total_proteins_fetched = int(total_proteins_fetched)

    return ExtractionStats(
        fetch_kind=fetch_kind,
        raw_records=raw_records,
        records_with_accession=records_with_accession,
        unique_accessions=len(seen),
        duplicate_records_skipped=records_with_accession - len(seen),
        proteins_reported=proteins_reported,
        total_proteins_fetched=total_proteins_fetched,
        architecture_count=len(data.get("architectures", [])) if fetch_kind == "dnaj" else None,
        total_architecture_instances=total_architecture_instances if fetch_kind == "dnaj" else None,
    )


def validate_extraction(stats: ExtractionStats) -> list[str]:
    """Return warnings when extraction counts disagree with fetch metadata."""
    warnings: list[str] = []

    if stats.fetch_kind == "dnak":
        if stats.total_proteins_fetched is not None and stats.raw_records != stats.total_proteins_fetched:
            warnings.append(
                f"DnaK JSON has {stats.raw_records} protein records but "
                f"total_proteins_fetched={stats.total_proteins_fetched}.",
            )
        if stats.unique_accessions < stats.raw_records:
            warnings.append(
                f"DnaK duplicate accessions in fetch output: {stats.raw_records - stats.unique_accessions} "
                f"duplicate record(s) collapsed to unique IDs.",
            )

    if stats.fetch_kind == "dnaj":
        if stats.total_architecture_instances is not None and stats.raw_records != stats.total_architecture_instances:
            warnings.append(
                f"DnaJ architecture protein instance count mismatch: "
                f"counted {stats.raw_records}, expected {stats.total_architecture_instances}.",
            )
        if stats.duplicate_records_skipped > 0:
            warnings.append(
                f"DnaJ fetch has {stats.duplicate_records_skipped} duplicate accession instance(s) "
                f"across architectures; Rockfish queue uses {stats.unique_accessions} unique ID(s).",
            )
        if stats.total_proteins_fetched is not None and stats.raw_records != stats.total_proteins_fetched:
            warnings.append(
                f"DnaJ JSON reports total_proteins_fetched={stats.total_proteins_fetched} "
                f"but counted {stats.raw_records} architecture protein instance(s).",
            )

    if stats.records_with_accession < stats.raw_records:
        warnings.append(
            f"{stats.raw_records - stats.records_with_accession} protein record(s) missing accession metadata.",
        )

    return warnings


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract UniProt accessions from InterPro fetch JSON for AlphaFoldFetch.",
    )
    parser.add_argument("json_file", type=Path, help="InterPro fetch output JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write one accession per line to this file (default: stdout)",
    )
    args = parser.parse_args()

    if not args.json_file.is_file():
        parser.error(f"JSON file not found: {args.json_file}")

    try:
        data = json.loads(args.json_file.read_text())
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON in {args.json_file}: {exc.msg}")

    for warning in fetch_warnings(data):
        sys.stderr.write(f"WARNING: {warning}\n")

    stats = extraction_stats(data)
    for warning in validate_extraction(stats):
        sys.stderr.write(f"WARNING: {warning}\n")

    accessions = extract_accessions(data)

    if not accessions:
        sys.stderr.write("WARNING: No UniProt accessions found in input JSON.\n")

    if args.output is None:
        sys.stdout.write("\n".join(accessions))
        if accessions:
            sys.stdout.write("\n")
    else:
        args.output.write_text("\n".join(accessions) + ("\n" if accessions else ""))

    sys.stderr.write(f"Extracted {len(accessions)} unique accessions.\n")
    if stats.duplicate_records_skipped > 0:
        sys.stderr.write(
            f"Collapsed {stats.duplicate_records_skipped} duplicate accession instance(s) "
            f"from {stats.raw_records} raw record(s).\n",
        )


if __name__ == "__main__":
    main()
