"""Rockfish array-job queue helpers: dedupe accessions, validate fetch counts, write snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


def accession_from_log_line(line: str) -> str | None:
    r"""Return the accession from a queue/completion/failure log line.

    Failure logs may be ``accession`` or ``accession\treason_code\tdetail``.
    """
    stripped = line.strip()
    if not stripped:
        return None
    first_field = stripped.split("\t", maxsplit=1)[0].strip()
    return normalize_accession(first_field)


def load_accession_lines(path: Path) -> list[str]:
    """Load non-empty accession keys from a text file, preserving first-seen order."""
    if not path.is_file():
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        accession = accession_from_log_line(line)
        if accession and accession not in seen:
            seen.add(accession)
            ordered.append(accession)
    return ordered


def write_accession_lines(path: Path, accessions: list[str]) -> None:
    """Write one accession per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(accessions) + ("\n" if accessions else ""), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One reason-coded failure log entry."""

    accession: str
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class FailureSummary:
    """Aggregated counts from a Rockfish failure log."""

    total: int
    counts_by_reason: dict[str, int]
    records: tuple[FailureRecord, ...]

    def accessions_for_reason(self, reason_code: str) -> list[str]:
        """Return unique accessions matching ``reason_code`` (first-seen order)."""
        ordered: list[str] = []
        seen: set[str] = set()
        for record in self.records:
            if record.reason_code != reason_code:
                continue
            if record.accession in seen:
                continue
            seen.add(record.accession)
            ordered.append(record.accession)
        return ordered


def parse_failure_log_line(line: str) -> FailureRecord | None:
    r"""Parse ``accession`` or ``accession\treason_code\tdetail`` into a FailureRecord."""
    stripped = line.strip()
    if not stripped:
        return None
    accession_part, *extra = stripped.split("\t")
    accession = normalize_accession(accession_part)
    if not accession:
        return None
    reason_code = "unspecified"
    detail = ""
    if extra:
        reason_code = extra[0].strip() or "unspecified"
        if extra[1:]:
            detail = extra[1].strip()
    return FailureRecord(accession=accession, reason_code=reason_code, detail=detail)


def load_failure_records(path: Path) -> list[FailureRecord]:
    """Load failure records from a Rockfish failed_*.txt log."""
    if not path.is_file():
        return []
    records: list[FailureRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = parse_failure_log_line(line)
        if record is not None:
            records.append(record)
    return records


def summarize_failures(path: Path) -> FailureSummary:
    """Count failure reason codes (useful after a large Rockfish array run)."""
    records = load_failure_records(path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.reason_code] = counts.get(record.reason_code, 0) + 1
    # Stable descending-by-count, then reason name.
    ordered_counts = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return FailureSummary(total=len(records), counts_by_reason=ordered_counts, records=tuple(records))


def pending_accessions(
    input_file: Path,
    completion_log: Path,
    *,
    failed_log: Path | None = None,
    include_failed: bool = False,
    retry_failed_only: bool = False,
) -> list[str]:
    """Return accessions still needing work.

    By default excludes completed and failed IDs. Pass ``include_failed=True``
    (CLI ``--retry-failed``) to re-queue known failures. Pass
    ``retry_failed_only=True`` to queue only failed IDs that are not completed.
    """
    input_accessions = load_accession_lines(input_file)
    completed = set(load_accession_lines(completion_log))
    failed = set(load_accession_lines(failed_log)) if failed_log is not None else set()

    if retry_failed_only:
        return [accession for accession in input_accessions if accession in failed and accession not in completed]

    pending = [accession for accession in input_accessions if accession not in completed]
    if include_failed or not failed:
        return pending
    return [accession for accession in pending if accession not in failed]


def pdb_structure_path(accession: str, structures_dir: Path, model_version: str) -> Path | None:
    """Return path to AF PDB (.pdb.gz or .pdb) when present."""
    base = structures_dir / f"AF-{accession}-F1-model_v{model_version}"
    for suffix in (".pdb.gz", ".pdb"):
        candidate = Path(str(base) + suffix)
        if candidate.is_file():
            return candidate
    return None


def cif_structure_path(accession: str, structures_dir: Path, model_version: str) -> Path | None:
    """Return path to AF CIF (.cif.gz or .cif) when present."""
    base = structures_dir / f"AF-{accession}-F1-model_v{model_version}"
    for suffix in (".cif.gz", ".cif"):
        candidate = Path(str(base) + suffix)
        if candidate.is_file():
            return candidate
    return None


def filter_accessions_with_pdb(
    accessions: list[str],
    structures_dir: Path,
    model_version: str,
) -> tuple[list[str], list[str], list[str]]:
    """Split accessions into (has_pdb, cif_only, missing_any)."""
    with_pdb: list[str] = []
    cif_only: list[str] = []
    missing: list[str] = []
    for accession in accessions:
        if pdb_structure_path(accession, structures_dir, model_version) is not None:
            with_pdb.append(accession)
        elif cif_structure_path(accession, structures_dir, model_version) is not None:
            cif_only.append(accession)
        else:
            missing.append(accession)
    return with_pdb, cif_only, missing


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    """Counts reported when writing an array snapshot."""

    input_count: int
    completed_count: int
    failed_count: int
    pending_count: int
    snapshot_count: int
    skipped_no_pdb: int = 0
    skipped_cif_only: int = 0


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


def _run_snapshot_command(args: argparse.Namespace) -> SnapshotStats:
    if not args.input.is_file():
        msg = f"Input file not found: {args.input}"
        raise SystemExit(msg)

    input_accessions = load_accession_lines(args.input)
    completed = load_accession_lines(args.completed)
    failed_path: Path | None = args.failed
    failed = load_accession_lines(failed_path) if failed_path is not None else []

    pending = pending_accessions(
        args.input,
        args.completed,
        failed_log=failed_path,
        include_failed=args.retry_failed,
        retry_failed_only=args.retry_failed_only,
    )

    skipped_no_pdb = 0
    skipped_cif_only = 0
    if args.require_pdb:
        if args.structures_dir is None:
            msg = "--require-pdb needs --structures-dir"
            raise SystemExit(msg)
        with_pdb, cif_only, missing = filter_accessions_with_pdb(
            pending,
            args.structures_dir,
            str(args.model_version),
        )
        skipped_cif_only = len(cif_only)
        skipped_no_pdb = len(missing) + skipped_cif_only
        if args.skipped_output is not None:
            skipped_rows = [f"{accession}\tcif_only\tCIF present but PDB required" for accession in cif_only] + [
                f"{accession}\tmissing_any_structure\tNo AF PDB/CIF under structures dir" for accession in missing
            ]
            args.skipped_output.parent.mkdir(parents=True, exist_ok=True)
            args.skipped_output.write_text(
                "\n".join(skipped_rows) + ("\n" if skipped_rows else ""),
                encoding="utf-8",
            )
        pending = with_pdb

    if not pending:
        sys.stderr.write(
            f"Nothing to queue: input={len(input_accessions)} completed={len(completed)} "
            f"failed={len(failed)} pending=0 "
            f"skipped_no_pdb={skipped_no_pdb} (cif_only={skipped_cif_only}).\n",
        )
        raise SystemExit(2)

    snapshot_rows = write_array_snapshot(pending, args.output, limit=args.limit)
    stats = SnapshotStats(
        input_count=len(input_accessions),
        completed_count=len(completed),
        failed_count=len(failed),
        pending_count=len(pending),
        snapshot_count=len(snapshot_rows),
        skipped_no_pdb=skipped_no_pdb,
        skipped_cif_only=skipped_cif_only,
    )
    sys.stderr.write(
        f"Wrote snapshot {args.output} with {stats.snapshot_count} accession(s) "
        f"(input={stats.input_count} completed={stats.completed_count} "
        f"failed={stats.failed_count} pending={stats.pending_count} "
        f"skipped_no_pdb={stats.skipped_no_pdb} cif_only={stats.skipped_cif_only} "
        f"limit={args.limit}).\n",
    )
    return stats


def _run_dedupe_command(args: argparse.Namespace) -> None:
    if not args.accession_file.is_file():
        msg = f"Accession file not found: {args.accession_file}"
        raise SystemExit(msg)

    before = args.accession_file.read_text(encoding="utf-8").splitlines()
    accessions = load_accession_lines(args.accession_file)
    write_accession_lines(args.accession_file, accessions)
    removed = len([line for line in before if accession_from_log_line(line)]) - len(accessions)
    sys.stderr.write(
        f"Deduped {args.accession_file}: {len(accessions)} unique accession(s) "
        f"({removed} duplicate line(s) removed).\n",
    )


def _run_summarize_failures_command(args: argparse.Namespace) -> None:
    if not args.failed_log.is_file():
        msg = f"Failure log not found: {args.failed_log}"
        raise SystemExit(msg)

    summary = summarize_failures(args.failed_log)
    sys.stderr.write(f"Failure log: {args.failed_log}\n")
    sys.stderr.write(f"Total failure rows: {summary.total}\n")
    if not summary.counts_by_reason:
        sys.stderr.write("No failure rows to summarize.\n")
        return

    sys.stderr.write("Counts by reason_code:\n")
    for reason, count in summary.counts_by_reason.items():
        sys.stderr.write(f"  {reason}\t{count}\n")

    if args.json_output is not None:
        payload = {
            "failed_log": str(args.failed_log),
            "total": summary.total,
            "counts_by_reason": summary.counts_by_reason,
            "records": [
                {
                    "accession": record.accession,
                    "reason_code": record.reason_code,
                    "detail": record.detail,
                }
                for record in summary.records
            ],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        sys.stderr.write(f"Wrote JSON summary: {args.json_output}\n")

    if args.reason is not None:
        accessions = summary.accessions_for_reason(args.reason)
        if args.write_accessions is None:
            msg = "--reason requires --write-accessions"
            raise SystemExit(msg)
        write_accession_lines(args.write_accessions, accessions)
        sys.stderr.write(
            f"Wrote {len(accessions)} accession(s) with reason '{args.reason}' to {args.write_accessions}\n",
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
        help="Write a fixed array snapshot from incomplete minus completed/failed accessions",
    )
    snapshot.add_argument("--input", type=Path, required=True, help="incomplete_accessions.txt")
    snapshot.add_argument("--completed", type=Path, required=True, help="completion log file")
    snapshot.add_argument(
        "--failed",
        type=Path,
        default=None,
        help="Failure log (excluded by default unless --retry-failed)",
    )
    snapshot.add_argument(
        "--retry-failed",
        action="store_true",
        help="Include previously failed accessions in the pending queue",
    )
    snapshot.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Queue only failed accessions that are not completed",
    )
    snapshot.add_argument(
        "--require-pdb",
        action="store_true",
        help="Keep only accessions with resolvable AF PDB under --structures-dir",
    )
    snapshot.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help="AlphaFold structures directory (required with --require-pdb)",
    )
    snapshot.add_argument(
        "--model-version",
        default="6",
        help="AlphaFold model version used in AF-<acc>-F1-model_vN filenames",
    )
    snapshot.add_argument(
        "--skipped-output",
        type=Path,
        default=None,
        help="Write accessions skipped by --require-pdb (reason-coded lines)",
    )
    snapshot.add_argument("-o", "--output", type=Path, required=True, help="Snapshot output path")
    snapshot.add_argument("--limit", type=int, default=MAX_ARRAY_TASKS)

    dedupe = subparsers.add_parser(
        "dedupe-file",
        help="Dedupe an accession list file in place (first occurrence wins)",
    )
    dedupe.add_argument("accession_file", type=Path)

    summarize = subparsers.add_parser(
        "summarize-failures",
        help="Summarize reason-coded Rockfish failure logs (counts by reason_code)",
    )
    summarize.add_argument(
        "failed_log",
        type=Path,
        help="failed_pocket.txt / failed_accessions.txt (accession[\\treason[\\tdetail]])",
    )
    summarize.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of counts and records",
    )
    summarize.add_argument(
        "--reason",
        default=None,
        help="If set with --write-accessions, export accessions matching this reason_code",
    )
    summarize.add_argument(
        "--write-accessions",
        type=Path,
        default=None,
        help="Write accessions for --reason (one per line) for targeted retries",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        _run_prepare_command(args)
        return

    if args.command == "write-snapshot":
        if args.retry_failed and args.retry_failed_only:
            msg = "Use only one of --retry-failed or --retry-failed-only"
            raise SystemExit(msg)
        _run_snapshot_command(args)
        return

    if args.command == "dedupe-file":
        _run_dedupe_command(args)
        return

    if args.command == "summarize-failures":
        if args.write_accessions is not None and args.reason is None:
            msg = "--write-accessions requires --reason"
            raise SystemExit(msg)
        _run_summarize_failures_command(args)


def main_prepare() -> None:
    """Console entry for prepare-rockfish-accessions (prepare subcommand only)."""
    if len(sys.argv) > 1 and sys.argv[1] in {
        "prepare",
        "write-snapshot",
        "dedupe-file",
        "summarize-failures",
    }:
        main()
        return
    sys.argv.insert(1, "prepare")
    main()


def main_summarize_failures() -> None:
    """Console entry for summarize-rockfish-failures."""
    if len(sys.argv) > 1 and sys.argv[1] == "summarize-failures":
        main()
        return
    sys.argv.insert(1, "summarize-failures")
    main()


if __name__ == "__main__":
    main()
