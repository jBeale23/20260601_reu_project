"""Merge DnaK/DnaJ fetch JSON, pocket charge CSV, and JDP classification CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jdp_classifier.classify import JDP_DATA_COLUMNS, JDP_SOURCE_COLUMNS
from scripts.extract_uniprot_ids import fetch_warnings
from scripts.merge_features import (
    POCKET_DATA_COLUMNS,
    FetchRecord,
    is_charge_inversion_candidate,
    iter_fetch_records,
    load_pocket_table,
)

JoinMode = Literal["outer", "inner"]


@dataclass(frozen=True, slots=True)
class SourcePresence:
    """Whether each merge source is present for one accession row."""

    has_dnak: bool
    has_dnaj: bool
    has_pocket: bool
    has_jdp: bool


@dataclass(frozen=True, slots=True)
class MergeAllInputs:
    """Loaded tables for unified feature merge."""

    dnak_by_accession: dict[str, FetchRecord]
    dnaj_by_accession: dict[str, FetchRecord]
    pocket_by_accession: dict[str, dict[str, str]]
    jdp_by_accession: dict[str, dict[str, str]]
    jdp_identity_by_accession: dict[str, dict[str, str]]
    provided_sources: dict[str, bool]
    join: JoinMode = "outer"


IDENTITY_COLUMNS = [
    "accession",
    "protein_name",
    "protein_length",
    "source_database",
    "fetch_sources",
]

DNAJ_ARCHITECTURE_COLUMNS = [
    "dnaj_architecture_ida",
    "dnaj_architecture_ida_id",
    "dnaj_appears_in_architecture_count",
]

POCKET_OUTPUT_COLUMNS = [
    column if column != "quality_flags" else "pocket_quality_flags" for column in POCKET_DATA_COLUMNS
]

DERIVED_COLUMNS = [
    "has_pocket_charge",
    "has_jdp_classification",
    "charge_inversion_candidate",
]

OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    *DNAJ_ARCHITECTURE_COLUMNS,
    *POCKET_OUTPUT_COLUMNS,
    *JDP_DATA_COLUMNS,
    *DERIVED_COLUMNS,
]


def load_jdp_table(csv_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Index jdp_classifications.csv by accession.

    Returns:
        Tuple of (jdp metrics by accession, identity fields by accession).
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"JDP CSV has no header row: {csv_path}"
            raise ValueError(msg)
        missing = set(JDP_SOURCE_COLUMNS) - set(reader.fieldnames)
        if "accession" not in reader.fieldnames or missing:
            msg = f"JDP CSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        jdp_by_accession: dict[str, dict[str, str]] = {}
        identity_by_accession: dict[str, dict[str, str]] = {}
        for row in reader:
            accession = row.get("accession", "").strip()
            if not accession:
                continue
            jdp_by_accession[accession] = {
                output_column: row.get(source_column, "")
                for source_column, output_column in zip(
                    JDP_SOURCE_COLUMNS,
                    JDP_DATA_COLUMNS,
                    strict=True,
                )
            }
            identity_by_accession[accession] = {
                "protein_name": row.get("protein_name", ""),
                "protein_length": row.get("protein_length", ""),
                "source_database": row.get("source_database", ""),
            }
    return jdp_by_accession, identity_by_accession


def _records_by_accession(records: list[FetchRecord]) -> dict[str, FetchRecord]:
    return {record.accession: record for record in records}


def _identity_from_record(record: FetchRecord) -> dict[str, str]:
    return {
        "protein_name": record.protein_name,
        "protein_length": record.protein_length,
        "source_database": record.source_database,
    }


def _identity_from_jdp_identity(identity: dict[str, str]) -> dict[str, str]:
    return {
        "protein_name": identity.get("protein_name", ""),
        "protein_length": identity.get("protein_length", ""),
        "source_database": identity.get("source_database", ""),
    }


def _pocket_output_row(pocket_row: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for column in POCKET_DATA_COLUMNS:
        output_column = "pocket_quality_flags" if column == "quality_flags" else column
        output[output_column] = pocket_row.get(column, "")
    return output


def _empty_pocket_row() -> dict[str, str]:
    return dict.fromkeys(POCKET_OUTPUT_COLUMNS, "")


def _empty_jdp_row() -> dict[str, str]:
    return dict.fromkeys(JDP_DATA_COLUMNS, "")


def _fetch_sources_for_accession(*, has_dnak: bool, has_dnaj: bool) -> str:
    sources: list[str] = []
    if has_dnak:
        sources.append("dnak")
    if has_dnaj:
        sources.append("dnaj")
    return ";".join(sources)


def _passes_inner_join(
    *,
    provided_sources: dict[str, bool],
    presence: SourcePresence,
) -> bool:
    presence_by_source = {
        "dnak": presence.has_dnak,
        "dnaj": presence.has_dnaj,
        "pocket": presence.has_pocket,
        "jdp": presence.has_jdp,
    }
    return all(presence_by_source[source] for source, is_provided in provided_sources.items() if is_provided)


def merge_all_features(inputs: MergeAllInputs) -> list[dict[str, str]]:
    """Build unified feature rows keyed by accession."""
    all_accessions = sorted(
        set(inputs.dnak_by_accession)
        | set(inputs.dnaj_by_accession)
        | set(inputs.pocket_by_accession)
        | set(inputs.jdp_by_accession),
    )

    merged_rows: list[dict[str, str]] = []
    for accession in all_accessions:
        dnak_record = inputs.dnak_by_accession.get(accession)
        dnaj_record = inputs.dnaj_by_accession.get(accession)
        pocket_row = inputs.pocket_by_accession.get(accession)
        jdp_row = inputs.jdp_by_accession.get(accession)

        presence = SourcePresence(
            has_dnak=dnak_record is not None,
            has_dnaj=dnaj_record is not None,
            has_pocket=pocket_row is not None,
            has_jdp=jdp_row is not None,
        )

        if inputs.join == "inner" and not _passes_inner_join(
            provided_sources=inputs.provided_sources,
            presence=presence,
        ):
            continue

        identity = {"protein_name": "", "protein_length": "", "source_database": ""}
        if dnak_record is not None:
            identity = _identity_from_record(dnak_record)
        elif dnaj_record is not None:
            identity = _identity_from_record(dnaj_record)
        elif accession in inputs.jdp_identity_by_accession:
            identity = _identity_from_jdp_identity(inputs.jdp_identity_by_accession[accession])

        row: dict[str, str] = {
            "accession": accession,
            **identity,
            "fetch_sources": _fetch_sources_for_accession(
                has_dnak=presence.has_dnak,
                has_dnaj=presence.has_dnaj,
            ),
            "dnaj_architecture_ida": dnaj_record.architecture_ida if dnaj_record else "",
            "dnaj_architecture_ida_id": dnaj_record.architecture_ida_id if dnaj_record else "",
            "dnaj_appears_in_architecture_count": (dnaj_record.appears_in_architecture_count if dnaj_record else ""),
            "has_pocket_charge": "true" if presence.has_pocket else "false",
            "has_jdp_classification": "true" if presence.has_jdp else "false",
            "charge_inversion_candidate": "false",
        }

        if presence.has_pocket and pocket_row is not None:
            row.update(_pocket_output_row(pocket_row))
            row["charge_inversion_candidate"] = (
                "true" if is_charge_inversion_candidate(pocket_row.get("quality_flags", "")) else "false"
            )
        else:
            row.update(_empty_pocket_row())

        if presence.has_jdp and jdp_row is not None:
            row.update(jdp_row)
        else:
            row.update(_empty_jdp_row())

        merged_rows.append(row)

    return merged_rows


def write_merged_all_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write unified feature rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _load_fetch_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc.msg}"
        raise ValueError(msg) from exc


def _load_merge_inputs(args: argparse.Namespace) -> MergeAllInputs:
    dnak_by_accession: dict[str, FetchRecord] = {}
    dnaj_by_accession: dict[str, FetchRecord] = {}
    pocket_by_accession: dict[str, dict[str, str]] = {}
    jdp_by_accession: dict[str, dict[str, str]] = {}
    jdp_identity_by_accession: dict[str, dict[str, str]] = {}

    if args.dnak_json is not None:
        dnak_data = _load_fetch_json(args.dnak_json)
        for warning in fetch_warnings(dnak_data):
            sys.stderr.write(f"WARNING: {warning}\n")
        dnak_by_accession = _records_by_accession(iter_fetch_records(dnak_data))

    if args.dnaj_json is not None:
        dnaj_data = _load_fetch_json(args.dnaj_json)
        for warning in fetch_warnings(dnaj_data):
            sys.stderr.write(f"WARNING: {warning}\n")
        dnaj_by_accession = _records_by_accession(iter_fetch_records(dnaj_data, dnaj_rows="dedupe"))

    if args.pocket_csv is not None:
        pocket_by_accession = load_pocket_table(args.pocket_csv)

    if args.jdp_csv is not None:
        jdp_by_accession, jdp_identity_by_accession = load_jdp_table(args.jdp_csv)

    return MergeAllInputs(
        dnak_by_accession=dnak_by_accession,
        dnaj_by_accession=dnaj_by_accession,
        pocket_by_accession=pocket_by_accession,
        jdp_by_accession=jdp_by_accession,
        jdp_identity_by_accession=jdp_identity_by_accession,
        provided_sources={
            "dnak": args.dnak_json is not None,
            "dnaj": args.dnaj_json is not None,
            "pocket": args.pocket_csv is not None,
            "jdp": args.jdp_csv is not None,
        },
        join=args.join,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge DnaK/DnaJ fetch JSON, pocket charge CSV, and JDP classification CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output unified feature CSV path",
    )
    parser.add_argument("--dnak-json", type=Path, default=None, help="DnaK InterPro fetch JSON")
    parser.add_argument("--dnaj-json", type=Path, default=None, help="DnaJ architecture fetch JSON")
    parser.add_argument(
        "--pocket-csv",
        type=Path,
        default=None,
        help="pocket_charge_summary.csv from analyze-pocket-charge",
    )
    parser.add_argument(
        "--jdp-csv",
        type=Path,
        default=None,
        help="jdp_classifications.csv from classify-jdp",
    )
    parser.add_argument(
        "--join",
        choices=("outer", "inner"),
        default="outer",
        help="outer = union of accessions; inner = only rows present in every provided source",
    )
    args = parser.parse_args()

    if not any([args.dnak_json, args.dnaj_json, args.pocket_csv, args.jdp_csv]):
        parser.error("At least one input (--dnak-json, --dnaj-json, --pocket-csv, --jdp-csv) is required.")

    for path, label in [
        (args.dnak_json, "DnaK JSON"),
        (args.dnaj_json, "DnaJ JSON"),
        (args.pocket_csv, "pocket CSV"),
        (args.jdp_csv, "JDP CSV"),
    ]:
        if path is not None and not path.is_file():
            parser.error(f"{label} not found: {path}")

    try:
        merge_inputs = _load_merge_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))

    merged_rows = merge_all_features(merge_inputs)
    write_merged_all_csv(merged_rows, args.output)

    with_pocket = sum(1 for row in merged_rows if row["has_pocket_charge"] == "true")
    with_jdp = sum(1 for row in merged_rows if row["has_jdp_classification"] == "true")
    sys.stderr.write(
        f"Merged {len(merged_rows)} row(s): "
        f"{len(merge_inputs.dnak_by_accession)} DnaK fetch, {len(merge_inputs.dnaj_by_accession)} DnaJ fetch, "
        f"{with_pocket} with pocket data, {with_jdp} with JDP classification.\n",
    )
    sys.stderr.write(f"Wrote {args.output}\n")


if __name__ == "__main__":
    main()
