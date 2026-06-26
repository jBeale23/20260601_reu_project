"""Rockfish array-job queue helpers: dedupe accessions, validate fetch counts, write snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.extract_uniprot_ids import (
    ExtractionStats,
    extract_accessions,
    extraction_stats,
    fetch_warnings,
    normalize_accession,
    validate_extraction,
)

MAX_ARRAY_TASKS = 10_000


def extract_accessions_normalized(data: dict[str, Any]) -> list[str]:
    """Extract sorted unique accessions with whitespace normalization."""
    return extract_accessions(data)


def load_accession_lines(path: Path) -> list[str]:
    """Load non-empty accession lines from a text file, preserving first-seen order."""
    if not path.is_file():
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        accession = normalize_accession(line)
        if accession and accession not in seen:
            seen.add(accession)
            ordered.append(accession)
    return ordered


def write_accession_lines(path: Path, accessions: list[str]) -> None:
    """Write one accession per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(accessions) + ("\n" if accessions else ""), encoding="utf-8")


def pending_accessions(input_file: Path, completion_log: Path) -> list[str]:
    """Return input accessions not present in the completion log (both deduped)."""
    input_accessions = load_accession_lines(input_file)
    completed = set(load_accession_lines(completion_log))
    return [accession for accession in input_accessions if accession not in completed]


def write_array_snapshot(
    pending: list[str],
    output_path: Path,
    *,
    limit: int = MAX_ARRAY_TASKS,
) -> list[str]:
    """Write up to ``limit`` pending accessions for a fixed SLURM array snapshot."""
    snapshot = pending[:limit]
    write_accession_lines(output_path, snapshot)
    return snapshot


def prepare_from_fetch_json(
    json_file: Path,
    *,
    output_accessions: Path | None = None,
    wk_dir: Path | None = None,
) -> tuple[list[str], ExtractionStats]:
    """Extract unique accessions from fetch JSON and optionally install into WK_DIR."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    stats = extraction_stats(data)
    accessions = extract_accessions_normalized(data)

    if output_accessions is not None:
        write_accession_lines(output_accessions, accessions)

    if wk_dir is not None:
        wk_input = wk_dir / "incomplete_accessions.txt"
        write_accession_lines(wk_input, accessions)

    return accessions, stats


def _print_stats(stats: ExtractionStats, accessions: list[str]) -> None:
    sys.stderr.write(
        f"Fetch kind: {stats.fetch_kind}; "
        f"raw_records={stats.raw_records}; "
        f"unique_accessions={len(accessions)}; "
        f"duplicate_records_skipped={stats.duplicate_records_skipped}\n",
    )
    if stats.fetch_kind == "dnak" and stats.total_proteins_fetched is not None:
        sys.stderr.write(
            f"DnaK total_proteins_fetched={stats.total_proteins_fetched}; "
            f"proteins_reported={stats.proteins_reported}\n",
        )
    if stats.fetch_kind == "dnaj":
        sys.stderr.write(
            f"DnaJ architectures={stats.architecture_count}; "
            f"architecture_instances={stats.total_architecture_instances}; "
            f"total_proteins_fetched={stats.total_proteins_fetched}\n",
        )


def _run_prepare_command(args: argparse.Namespace) -> None:
    if not args.fetch_json.is_file():
        msg = f"Fetch JSON not found: {args.fetch_json}"
        raise SystemExit(msg)

    try:
        data = json.loads(args.fetch_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {args.fetch_json}: {exc.msg}"
        raise SystemExit(msg) from exc

    for warning in fetch_warnings(data):
        sys.stderr.write(f"WARNING: {warning}\n")

    accessions, stats = prepare_from_fetch_json(
        args.fetch_json,
        output_accessions=args.output,
        wk_dir=args.wk_dir,
    )
    for warning in validate_extraction(stats):
        sys.stderr.write(f"WARNING: {warning}\n")

    _print_stats(stats, accessions)
    if args.wk_dir is not None:
        sys.stderr.write(f"Wrote {args.wk_dir / 'incomplete_accessions.txt'}\n")
    sys.stderr.write(f"Wrote {args.output} ({len(accessions)} unique accessions).\n")


def _run_snapshot_command(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        msg = f"Input file not found: {args.input}"
        raise SystemExit(msg)

    pending = pending_accessions(args.input, args.completed)
    if not pending:
        raise SystemExit(2)

    snapshot_rows = write_array_snapshot(pending, args.output, limit=args.limit)
    sys.stderr.write(
        f"Wrote snapshot {args.output} with {len(snapshot_rows)} accession(s) "
        f"({len(pending)} pending total, limit {args.limit}).\n",
    )


def _run_dedupe_command(args: argparse.Namespace) -> None:
    if not args.accession_file.is_file():
        msg = f"Accession file not found: {args.accession_file}"
        raise SystemExit(msg)

    before = args.accession_file.read_text(encoding="utf-8").splitlines()
    accessions = load_accession_lines(args.accession_file)
    write_accession_lines(args.accession_file, accessions)
    removed = len([line for line in before if normalize_accession(line)]) - len(accessions)
    sys.stderr.write(
        f"Deduped {args.accession_file}: {len(accessions)} unique accession(s) "
        f"({removed} duplicate line(s) removed).\n",
    )


def main() -> None:
    """CLI entry point for Rockfish accession queue utilities."""
    parser = argparse.ArgumentParser(description="Prepare and validate Rockfish accession queue files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Extract unique accessions from fetch JSON and write deduped queue files",
    )
    prepare.add_argument("fetch_json", type=Path, help="DnaK or DnaJ InterPro fetch JSON")
    prepare.add_argument("-o", "--output", type=Path, default=Path("accessions.txt"))
    prepare.add_argument(
        "--wk-dir",
        type=Path,
        default=None,
        help="Also write deduped incomplete_accessions.txt into this Rockfish work directory",
    )

    snapshot = subparsers.add_parser(
        "write-snapshot",
        help="Write a fixed array snapshot from incomplete minus completed accessions",
    )
    snapshot.add_argument("--input", type=Path, required=True, help="incomplete_accessions.txt")
    snapshot.add_argument("--completed", type=Path, required=True, help="completion log file")
    snapshot.add_argument("-o", "--output", type=Path, required=True, help="Snapshot output path")
    snapshot.add_argument("--limit", type=int, default=MAX_ARRAY_TASKS)

    dedupe = subparsers.add_parser(
        "dedupe-file",
        help="Dedupe an accession list file in place (first occurrence wins)",
    )
    dedupe.add_argument("accession_file", type=Path)

    args = parser.parse_args()

    if args.command == "prepare":
        _run_prepare_command(args)
        return

    if args.command == "write-snapshot":
        _run_snapshot_command(args)
        return

    if args.command == "dedupe-file":
        _run_dedupe_command(args)


def main_prepare() -> None:
    """Console entry for prepare-rockfish-accessions (prepare subcommand only)."""
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare", "write-snapshot", "dedupe-file"}:
        main()
        return
    sys.argv.insert(1, "prepare")
    main()


if __name__ == "__main__":
    main()
