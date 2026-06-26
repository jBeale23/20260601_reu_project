"""Merge InterPro fetch JSON with pocket charge summary CSV by UniProt accession."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scripts.extract_uniprot_ids import fetch_warnings, normalize_accession

# Pocket metric columns from structure_analysis.analyze.CSV_COLUMNS (excluding accession).
POCKET_DATA_COLUMNS = [
    "confidence_tier",
    "mapping_confidence",
    "conservation_score",
    "mapping_mode",
    "contact_net_charge",
    "shell_net_charge",
    "delta_contact_net_charge",
    "delta_net_charge_vs_reference",
    "contact_mapping_fraction",
    "contact_sequence_identity",
    "mean_contact_ca_distance",
    "contact_drmsd",
    "n_contacts_within_5A",
    "sbd_alignment_coverage",
    "mean_plddt_pocket",
    "mapped_pocket_residues",
    "n_histidine",
    "net_charge_excluding_his",
    "exposed_net_charge",
    "charge_density",
    "quality_flags",
]

IDENTITY_COLUMNS = [
    "accession",
    "protein_name",
    "protein_length",
    "source_database",
    "fetch_source",
]

ARCHITECTURE_COLUMNS = [
    "architecture_ida",
    "architecture_ida_id",
    "appears_in_architecture_count",
]

DERIVED_COLUMNS = [
    "has_pocket_charge",
    "charge_inversion_candidate",
    "merge_warnings",
]

OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    *ARCHITECTURE_COLUMNS,
    *POCKET_DATA_COLUMNS,
    *DERIVED_COLUMNS,
]

JoinMode = Literal["left", "inner"]
DnajRowsMode = Literal["dedupe", "explode"]


@dataclass(frozen=True, slots=True)
class FetchRecord:
    """Normalized protein row from a DnaK or DnaJ fetch JSON file."""

    accession: str
    protein_name: str
    protein_length: str
    source_database: str
    fetch_source: Literal["dnak", "dnaj"]
    architecture_ida: str
    architecture_ida_id: str
    appears_in_architecture_count: str


def load_pocket_table(csv_path: Path) -> dict[str, dict[str, str]]:
    """Index pocket_charge_summary.csv by accession."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"Pocket CSV has no header row: {csv_path}"
            raise ValueError(msg)
        missing = set(POCKET_DATA_COLUMNS) - set(reader.fieldnames)
        if "accession" not in reader.fieldnames or missing:
            msg = f"Pocket CSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        pocket_by_accession: dict[str, dict[str, str]] = {}
        for row in reader:
            accession = normalize_accession(row.get("accession"))
            if not accession:
                continue
            pocket_by_accession[accession] = {column: row.get(column, "") for column in POCKET_DATA_COLUMNS}
    return pocket_by_accession


def _metadata_field(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value)


def _record_from_protein(
    protein: dict[str, Any],
    *,
    fetch_source: Literal["dnak", "dnaj"],
    architecture_ida: str = "",
    architecture_ida_id: str = "",
) -> FetchRecord | None:
    metadata = protein.get("metadata", {})
    accession = normalize_accession(metadata.get("accession"))
    if not accession:
        return None

    appears_count = protein.get("appears_in_architecture_count")
    return FetchRecord(
        accession=accession,
        protein_name=_metadata_field(metadata, "name"),
        protein_length=_metadata_field(metadata, "length"),
        source_database=_metadata_field(metadata, "source_database"),
        fetch_source=fetch_source,
        architecture_ida=architecture_ida,
        architecture_ida_id=architecture_ida_id,
        appears_in_architecture_count="" if appears_count is None else str(appears_count),
    )


def _iter_dnaj_exploded(data: dict[str, Any]) -> list[FetchRecord]:
    records: list[FetchRecord] = []
    for architecture in data.get("architectures", []):
        ida = str(architecture.get("ida", ""))
        ida_id = str(architecture.get("ida_id", ""))
        for protein in architecture.get("proteins", []):
            record = _record_from_protein(
                protein,
                fetch_source="dnaj",
                architecture_ida=ida,
                architecture_ida_id=ida_id,
            )
            if record is not None:
                records.append(record)
    return records


def _iter_dnaj_deduped(data: dict[str, Any]) -> list[FetchRecord]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "protein": None,
            "idas": set(),
            "ida_ids": set(),
        },
    )

    for architecture in data.get("architectures", []):
        ida = str(architecture.get("ida", ""))
        ida_id = str(architecture.get("ida_id", ""))
        for protein in architecture.get("proteins", []):
            metadata = protein.get("metadata", {})
            accession = normalize_accession(metadata.get("accession"))
            if not accession:
                continue
            entry = grouped[accession]
            if entry["protein"] is None:
                entry["protein"] = protein
            if ida:
                entry["idas"].add(ida)
            if ida_id:
                entry["ida_ids"].add(ida_id)

    records: list[FetchRecord] = []
    for accession in sorted(grouped):
        entry = grouped[accession]
        protein = entry["protein"]
        if protein is None:
            msg = f"Missing protein payload for accession {accession}"
            raise ValueError(msg)
        record = _record_from_protein(
            protein,
            fetch_source="dnaj",
            architecture_ida=";".join(sorted(entry["idas"])),
            architecture_ida_id=";".join(sorted(entry["ida_ids"])),
        )
        if record is None:
            msg = f"Could not build fetch record for accession {accession}"
            raise ValueError(msg)
        records.append(record)
    return records


def iter_fetch_records(
    data: dict[str, Any],
    *,
    dnaj_rows: DnajRowsMode = "dedupe",
) -> list[FetchRecord]:
    """Return normalized records from a DnaK or DnaJ fetch JSON file."""
    if data.get("architectures") is not None:
        if dnaj_rows == "explode":
            return _iter_dnaj_exploded(data)
        return _iter_dnaj_deduped(data)

    records: list[FetchRecord] = []
    for protein in data.get("proteins", []):
        record = _record_from_protein(protein, fetch_source="dnak")
        if record is not None:
            records.append(record)
    return records


def is_charge_inversion_candidate(quality_flags: str) -> bool:
    """Return True when pocket quality_flags contains charge_inversion_candidate."""
    flags = {flag.strip() for flag in quality_flags.split(";") if flag.strip()}
    return "charge_inversion_candidate" in flags


def merge_records(
    fetch_records: list[FetchRecord],
    pocket_by_accession: dict[str, dict[str, str]],
    *,
    join: JoinMode = "left",
) -> tuple[list[dict[str, str]], int]:
    """Join fetch records with pocket metrics.

    Returns:
        Tuple of merged rows and count of pocket CSV accessions not present in fetch.
    """
    fetch_accessions = {record.accession for record in fetch_records}
    orphan_count = sum(1 for accession in pocket_by_accession if accession not in fetch_accessions)

    merged_rows: list[dict[str, str]] = []
    for record in fetch_records:
        pocket_row = pocket_by_accession.get(record.accession)
        has_pocket = pocket_row is not None
        if join == "inner" and not has_pocket:
            continue

        row: dict[str, str] = {
            "accession": record.accession,
            "protein_name": record.protein_name,
            "protein_length": record.protein_length,
            "source_database": record.source_database,
            "fetch_source": record.fetch_source,
            "architecture_ida": record.architecture_ida,
            "architecture_ida_id": record.architecture_ida_id,
            "appears_in_architecture_count": record.appears_in_architecture_count,
            "has_pocket_charge": "true" if has_pocket else "false",
            "charge_inversion_candidate": "false",
            "merge_warnings": "",
        }

        if has_pocket and pocket_row is not None:
            row.update(pocket_row)
            row["charge_inversion_candidate"] = (
                "true" if is_charge_inversion_candidate(pocket_row.get("quality_flags", "")) else "false"
            )
        else:
            for column in POCKET_DATA_COLUMNS:
                row[column] = ""

        merged_rows.append(row)

    return merged_rows, orphan_count


def write_merged_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write merged feature rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge InterPro fetch JSON with pocket charge summary CSV by accession.",
    )
    parser.add_argument("fetch_json", type=Path, help="InterPro fetch output JSON file")
    parser.add_argument(
        "--pocket-csv",
        type=Path,
        required=True,
        help="pocket_charge_summary.csv from analyze-pocket-charge --batch",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output merged feature CSV path",
    )
    parser.add_argument(
        "--join",
        choices=("left", "inner"),
        default="left",
        help="Join mode: left keeps all fetch proteins; inner keeps only rows with pocket data",
    )
    parser.add_argument(
        "--dnaj-rows",
        choices=("dedupe", "explode"),
        default="dedupe",
        help="DnaJ row layout: dedupe one row per accession, explode one row per architecture",
    )
    args = parser.parse_args()

    if not args.fetch_json.is_file():
        parser.error(f"Fetch JSON not found: {args.fetch_json}")
    if not args.pocket_csv.is_file():
        parser.error(f"Pocket CSV not found: {args.pocket_csv}")

    try:
        fetch_data = json.loads(args.fetch_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON in {args.fetch_json}: {exc.msg}")

    for warning in fetch_warnings(fetch_data):
        sys.stderr.write(f"WARNING: {warning}\n")

    try:
        pocket_by_accession = load_pocket_table(args.pocket_csv)
    except ValueError as exc:
        parser.error(str(exc))

    fetch_records = iter_fetch_records(fetch_data, dnaj_rows=args.dnaj_rows)
    if not fetch_records:
        sys.stderr.write("WARNING: No fetch proteins found in input JSON.\n")

    merged_rows, orphan_count = merge_records(
        fetch_records,
        pocket_by_accession,
        join=args.join,
    )
    write_merged_csv(merged_rows, args.output)

    with_pocket = sum(1 for row in merged_rows if row["has_pocket_charge"] == "true")
    sys.stderr.write(
        f"Merged {len(merged_rows)} rows "
        f"({len(fetch_records)} fetch proteins, {with_pocket} with pocket data, "
        f"{orphan_count} orphan pocket CSV accessions).\n",
    )
    sys.stderr.write(f"Wrote {args.output}\n")


if __name__ == "__main__":
    main()
