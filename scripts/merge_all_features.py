"""Merge DnaK/DnaJ fetch JSON, pocket charge CSV, and JDP classification CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jdp_classifier.classify import JDP_DATA_COLUMNS, JDP_SOURCE_COLUMNS
from scripts.chaperone_profiles import ChaperonePresence, derive_chaperone_classification_fields
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
    has_motif: bool
    has_domain: bool = False


@dataclass(frozen=True, slots=True)
class MergeAllInputs:
    """Loaded tables for unified feature merge."""

    dnak_by_accession: dict[str, FetchRecord]
    dnaj_by_accession: dict[str, FetchRecord]
    pocket_by_accession: dict[str, dict[str, str]]
    jdp_by_accession: dict[str, dict[str, str]]
    jdp_identity_by_accession: dict[str, dict[str, str]]
    motif_by_accession: dict[str, dict[str, str]]
    provided_sources: dict[str, bool]
    domain_by_accession: dict[str, dict[str, str]] = field(default_factory=dict)
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

MOTIF_DATA_COLUMNS = [
    "motif_domain_families",
    "motif_block_grammars",
    "motif_best_window_lengths",
    "motif_net_charges",
    "motif_window_net_charges",
]

# Columns taken from domain_layout_features.csv (analyze-domain-layout), prefixed to keep
# them distinct from the v1 classifier columns.
DOMAIN_SOURCE_COLUMNS = [
    "n_structured_domains",
    "domain_family_layout",
    "j_domain_position",
    "has_hpd",
    "has_gf_rich_region",
    "idr_fraction",
    "disorder_backend",
    "n_msa_regions",
    "n_shark_regions",
    "shark_backend",
    "shark_best_reference_class",
    "shark_best_similarity",
    "layout_predicted_class",
    "layout_predicted_subclass",
    "layout_class_confidence",
    "layout_novelty_score",
    "novel_class_candidate",
    "layout_evidence_tags",
]
DOMAIN_DATA_COLUMNS = [f"domain_{column}" for column in DOMAIN_SOURCE_COLUMNS]

DERIVED_COLUMNS = [
    "has_pocket_charge",
    "has_jdp_classification",
    "has_motif_features",
    "has_domain_layout",
    "charge_inversion_candidate",
    "novel_class_candidate",
    "chaperone_system_membership",
    "unified_confidence_tier",
    "classification_tags",
]

OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    *DNAJ_ARCHITECTURE_COLUMNS,
    *POCKET_OUTPUT_COLUMNS,
    *JDP_DATA_COLUMNS,
    *MOTIF_DATA_COLUMNS,
    *DOMAIN_DATA_COLUMNS,
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


def load_motif_table(csv_path: Path) -> dict[str, dict[str, str]]:
    """Aggregate motif_accession_features.csv rows by accession."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "accession" not in reader.fieldnames:
            msg = f"Motif CSV missing accession column: {csv_path}"
            raise ValueError(msg)

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            grouped.setdefault(accession, []).append(row)

    aggregated: dict[str, dict[str, str]] = {}
    for accession, rows in grouped.items():
        aggregated[accession] = {
            "motif_domain_families": ";".join(row.get("domain_family", "") for row in rows),
            "motif_block_grammars": ";".join(row.get("block_grammar", "") for row in rows),
            "motif_best_window_lengths": ";".join(str(row.get("best_window_length", "")) for row in rows),
            "motif_net_charges": ";".join(str(row.get("net_charge", "")) for row in rows),
            "motif_window_net_charges": ";".join(str(row.get("window_net_charge", "")) for row in rows),
        }
    return aggregated


def load_domain_layout_table(csv_path: Path) -> dict[str, dict[str, str]]:
    """Index domain_layout_features.csv by accession, prefixing columns with ``domain_``.

    Raises:
        ValueError: If the CSV has no header or is missing required columns.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"Domain layout CSV has no header row: {csv_path}"
            raise ValueError(msg)
        missing = set(DOMAIN_SOURCE_COLUMNS) - set(reader.fieldnames)
        if "accession" not in reader.fieldnames or missing:
            msg = f"Domain layout CSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        by_accession: dict[str, dict[str, str]] = {}
        for row in reader:
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            by_accession[accession] = {
                output_column: row.get(source_column, "")
                for source_column, output_column in zip(DOMAIN_SOURCE_COLUMNS, DOMAIN_DATA_COLUMNS, strict=True)
            }
    return by_accession


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


def _empty_motif_row() -> dict[str, str]:
    return dict.fromkeys(MOTIF_DATA_COLUMNS, "")


def _empty_domain_row() -> dict[str, str]:
    return dict.fromkeys(DOMAIN_DATA_COLUMNS, "")


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
        "motif": presence.has_motif,
        "domain": presence.has_domain,
    }
    return all(presence_by_source[source] for source, is_provided in provided_sources.items() if is_provided)


def merge_all_features(inputs: MergeAllInputs) -> list[dict[str, str]]:
    """Build unified feature rows keyed by accession."""
    all_accessions = sorted(
        set(inputs.dnak_by_accession)
        | set(inputs.dnaj_by_accession)
        | set(inputs.pocket_by_accession)
        | set(inputs.jdp_by_accession)
        | set(inputs.motif_by_accession)
        | set(inputs.domain_by_accession),
    )

    merged_rows: list[dict[str, str]] = []
    for accession in all_accessions:
        row = _merged_row(accession, inputs)
        if row is not None:
            merged_rows.append(row)

    return merged_rows


def _merged_row(accession: str, inputs: MergeAllInputs) -> dict[str, str] | None:
    """Build one unified row, or ``None`` when an inner join excludes the accession."""
    dnak_record = inputs.dnak_by_accession.get(accession)
    dnaj_record = inputs.dnaj_by_accession.get(accession)
    pocket_row = inputs.pocket_by_accession.get(accession)
    jdp_row = inputs.jdp_by_accession.get(accession)
    motif_row = inputs.motif_by_accession.get(accession)
    domain_row = inputs.domain_by_accession.get(accession)

    presence = SourcePresence(
        has_dnak=dnak_record is not None,
        has_dnaj=dnaj_record is not None,
        has_pocket=pocket_row is not None,
        has_jdp=jdp_row is not None,
        has_motif=motif_row is not None,
        has_domain=domain_row is not None,
    )

    if inputs.join == "inner" and not _passes_inner_join(
        provided_sources=inputs.provided_sources,
        presence=presence,
    ):
        return None

    row: dict[str, str] = {
        "accession": accession,
        **_resolve_identity(accession, inputs, dnak_record=dnak_record, dnaj_record=dnaj_record),
        "fetch_sources": _fetch_sources_for_accession(
            has_dnak=presence.has_dnak,
            has_dnaj=presence.has_dnaj,
        ),
        "dnaj_architecture_ida": dnaj_record.architecture_ida if dnaj_record else "",
        "dnaj_architecture_ida_id": dnaj_record.architecture_ida_id if dnaj_record else "",
        "dnaj_appears_in_architecture_count": (dnaj_record.appears_in_architecture_count if dnaj_record else ""),
        "has_pocket_charge": "true" if presence.has_pocket else "false",
        "has_jdp_classification": "true" if presence.has_jdp else "false",
        "has_motif_features": "true" if presence.has_motif else "false",
        "has_domain_layout": "true" if presence.has_domain else "false",
        "charge_inversion_candidate": "false",
        "novel_class_candidate": "false",
    }

    if pocket_row is not None:
        row.update(_pocket_output_row(pocket_row))
        row["charge_inversion_candidate"] = (
            "true" if is_charge_inversion_candidate(pocket_row.get("quality_flags", "")) else "false"
        )
    else:
        row.update(_empty_pocket_row())

    row.update(jdp_row if jdp_row is not None else _empty_jdp_row())
    row.update(motif_row if motif_row is not None else _empty_motif_row())

    if domain_row is not None:
        row.update(domain_row)
        row["novel_class_candidate"] = domain_row.get("domain_novel_class_candidate", "false") or "false"
    else:
        row.update(_empty_domain_row())

    row.update(
        derive_chaperone_classification_fields(
            presence=ChaperonePresence(
                has_dnak=presence.has_dnak,
                has_dnaj=presence.has_dnaj,
                has_jdp=presence.has_jdp,
                has_pocket=presence.has_pocket,
                has_motif=presence.has_motif,
                charge_inversion_candidate=row["charge_inversion_candidate"] == "true",
                has_domain_layout=presence.has_domain,
                novel_class_candidate=row["novel_class_candidate"] == "true",
            ),
            jdp_row={column: row.get(column, "") for column in JDP_DATA_COLUMNS},
            pocket_row={column: row.get(column, "") for column in POCKET_OUTPUT_COLUMNS},
            domain_row={column: row.get(column, "") for column in DOMAIN_DATA_COLUMNS},
        ),
    )
    return row


def _resolve_identity(
    accession: str,
    inputs: MergeAllInputs,
    *,
    dnak_record: FetchRecord | None,
    dnaj_record: FetchRecord | None,
) -> dict[str, str]:
    """Resolve identity fields in order: DnaK fetch, DnaJ fetch, JDP CSV."""
    if dnak_record is not None:
        return _identity_from_record(dnak_record)
    if dnaj_record is not None:
        return _identity_from_record(dnaj_record)
    if accession in inputs.jdp_identity_by_accession:
        return _identity_from_jdp_identity(inputs.jdp_identity_by_accession[accession])
    return {"protein_name": "", "protein_length": "", "source_database": ""}


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
    motif_by_accession: dict[str, dict[str, str]] = {}

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

    if args.motif_csv is not None:
        motif_by_accession = load_motif_table(args.motif_csv)

    domain_by_accession: dict[str, dict[str, str]] = {}
    if args.domain_csv is not None:
        domain_by_accession = load_domain_layout_table(args.domain_csv)

    return MergeAllInputs(
        dnak_by_accession=dnak_by_accession,
        dnaj_by_accession=dnaj_by_accession,
        pocket_by_accession=pocket_by_accession,
        jdp_by_accession=jdp_by_accession,
        jdp_identity_by_accession=jdp_identity_by_accession,
        motif_by_accession=motif_by_accession,
        domain_by_accession=domain_by_accession,
        provided_sources={
            "dnak": args.dnak_json is not None,
            "dnaj": args.dnaj_json is not None,
            "pocket": args.pocket_csv is not None,
            "jdp": args.jdp_csv is not None,
            "motif": args.motif_csv is not None,
            "domain": args.domain_csv is not None,
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
        "--motif-csv",
        type=Path,
        default=None,
        help="motif_accession_features.csv from analyze-motif-conservation",
    )
    parser.add_argument(
        "--domain-csv",
        type=Path,
        default=None,
        help="domain_layout_features.csv from analyze-domain-layout",
    )
    parser.add_argument(
        "--join",
        choices=("outer", "inner"),
        default="outer",
        help="outer = union of accessions; inner = only rows present in every provided source",
    )
    args = parser.parse_args()

    if not any([args.dnak_json, args.dnaj_json, args.pocket_csv, args.jdp_csv, args.motif_csv, args.domain_csv]):
        parser.error(
            "At least one input (--dnak-json, --dnaj-json, --pocket-csv, --jdp-csv, "
            "--motif-csv, --domain-csv) is required.",
        )

    for path, label in [
        (args.dnak_json, "DnaK JSON"),
        (args.dnaj_json, "DnaJ JSON"),
        (args.pocket_csv, "pocket CSV"),
        (args.jdp_csv, "JDP CSV"),
        (args.motif_csv, "motif CSV"),
        (args.domain_csv, "domain layout CSV"),
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
    with_motif = sum(1 for row in merged_rows if row["has_motif_features"] == "true")
    with_domain = sum(1 for row in merged_rows if row["has_domain_layout"] == "true")
    novel = sum(1 for row in merged_rows if row["novel_class_candidate"] == "true")
    sys.stderr.write(
        f"Merged {len(merged_rows)} row(s): "
        f"{len(merge_inputs.dnak_by_accession)} DnaK fetch, {len(merge_inputs.dnaj_by_accession)} DnaJ fetch, "
        f"{with_pocket} with pocket data, {with_jdp} with JDP classification, "
        f"{with_motif} with motif features, {with_domain} with domain layout "
        f"({novel} novel-class candidate(s)).\n",
    )
    sys.stderr.write(f"Wrote {args.output}\n")


if __name__ == "__main__":
    main()
