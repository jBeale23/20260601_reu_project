"""Tests for scripts/rockfish_queue.py."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.capture import CaptureFixture

from scripts.extract_uniprot_ids import extract_accessions
from scripts.rockfish_queue import (
    accession_from_log_line,
    extraction_stats,
    filter_accessions_with_pdb,
    load_accession_lines,
    parse_failure_log_line,
    pending_accessions,
    prepare_from_fetch_json,
    summarize_failures,
    validate_extraction,
    write_array_snapshot,
)
from scripts.rockfish_queue import (
    main as rockfish_main,
)


def test_extraction_stats_dnaj_counts_duplicates() -> None:
    """DnaJ duplicate architecture instances are counted separately from unique IDs."""
    data = {
        "total_proteins_fetched": 4,
        "architectures": [
            {"proteins": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P2"}}]},
            {"proteins": [{"metadata": {"accession": "P2"}}, {"metadata": {"accession": "P3"}}]},
        ],
    }
    stats = extraction_stats(data)
    assert stats.fetch_kind == "dnaj"
    assert stats.raw_records == 4
    assert stats.unique_accessions == 3
    assert stats.duplicate_records_skipped == 1
    assert extract_accessions(data) == ["P1", "P2", "P3"]


def test_extraction_stats_dnak_unique_matches_records() -> None:
    """DnaK records without duplicates yield matching raw and unique counts."""
    data = {
        "total_proteins_fetched": 2,
        "proteins": [
            {"metadata": {"accession": "P0A6Y8"}},
            {"metadata": {"accession": " P08113 "}},
        ],
    }
    stats = extraction_stats(data)
    assert stats.fetch_kind == "dnak"
    assert stats.raw_records == 2
    assert stats.unique_accessions == 2
    assert extract_accessions(data) == ["P08113", "P0A6Y8"]


def test_validate_extraction_warns_on_dnaj_duplicates() -> None:
    """Validation explains expected DnaJ duplicate collapse."""
    stats = extraction_stats(
        {
            "architectures": [
                {"proteins": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P1"}}]},
            ],
        },
    )
    warnings = validate_extraction(stats)
    assert any("duplicate accession instance" in warning for warning in warnings)


def test_load_accession_lines_dedupes_preserving_order(tmp_path: Path) -> None:
    """Accession files are deduped with first occurrence kept."""
    path = tmp_path / "accessions.txt"
    path.write_text("P1\nP2\nP1\n P3 \n\n", encoding="utf-8")
    assert load_accession_lines(path) == ["P1", "P2", "P3"]


def test_accession_from_log_line_parses_reason_coded_failures() -> None:
    """Failure log lines keep only the accession key."""
    assert accession_from_log_line("P1\tcif_only\tCIF present but PDB required") == "P1"
    assert accession_from_log_line("P2") == "P2"


def test_pending_accessions_excludes_completed(tmp_path: Path) -> None:
    """Pending queue excludes accessions already in the completion log."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    input_file.write_text("P1\nP2\nP3\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")
    assert pending_accessions(input_file, completion_log) == ["P1", "P3"]


def test_pending_accessions_excludes_failed_by_default(tmp_path: Path) -> None:
    """Failed accessions are excluded unless retry is requested."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    failed_log = tmp_path / "failed.txt"
    input_file.write_text("P1\nP2\nP3\nP4\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")
    failed_log.write_text("P3\tcif_only\tdetail\n", encoding="utf-8")
    assert pending_accessions(input_file, completion_log, failed_log=failed_log) == ["P1", "P4"]
    assert pending_accessions(
        input_file,
        completion_log,
        failed_log=failed_log,
        include_failed=True,
    ) == ["P1", "P3", "P4"]
    assert pending_accessions(
        input_file,
        completion_log,
        failed_log=failed_log,
        retry_failed_only=True,
    ) == ["P3"]


def test_filter_accessions_with_pdb_separates_cif_only(tmp_path: Path) -> None:
    """PDB preflight distinguishes PDB, CIF-only, and missing structures."""
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "AF-P1-F1-model_v6.pdb.gz").write_text("pdb", encoding="utf-8")
    (structures / "AF-P2-F1-model_v6.cif.gz").write_text("cif", encoding="utf-8")
    with_pdb, cif_only, missing = filter_accessions_with_pdb(["P1", "P2", "P3"], structures, "6")
    assert with_pdb == ["P1"]
    assert cif_only == ["P2"]
    assert missing == ["P3"]


def test_write_array_snapshot_fixed_mapping(tmp_path: Path) -> None:
    """Snapshot line N is stable for the life of the array job."""
    output = tmp_path / "snapshot.txt"
    pending = ["A", "B", "C", "D"]
    snapshot = write_array_snapshot(pending, output, limit=2)
    assert snapshot == ["A", "B"]
    assert load_accession_lines(output) == ["A", "B"]


def test_prepare_from_fetch_json_writes_wk_dir(tmp_path: Path) -> None:
    """Prepare writes deduped accessions into the Rockfish work directory."""
    fetch_json = tmp_path / "dnak.json"
    wk_dir = tmp_path / "wk"
    fetch_json.write_text(
        json.dumps(
            {
                "total_proteins_fetched": 2,
                "proteins": [
                    {"metadata": {"accession": "P1"}},
                    {"metadata": {"accession": "P2"}},
                ],
            },
        ),
        encoding="utf-8",
    )
    accessions, stats = prepare_from_fetch_json(fetch_json, wk_dir=wk_dir)
    assert accessions == ["P1", "P2"]
    assert stats.unique_accessions == 2
    assert (wk_dir / "incomplete_accessions.txt").read_text(encoding="utf-8") == "P1\nP2\n"


def test_dedupe_file_cli(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """dedupe-file collapses duplicate accession lines in place."""
    accession_file = tmp_path / "accessions.txt"
    accession_file.write_text("P1\nP2\nP1\n P3 \n\n", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["rockfish_queue.py", "dedupe-file", str(accession_file)],
    ):
        rockfish_main()

    assert accession_file.read_text(encoding="utf-8") == "P1\nP2\nP3\n"
    captured = capsys.readouterr()
    assert "3 unique accession(s)" in captured.err
    assert "1 duplicate line(s) removed" in captured.err


def test_write_snapshot_cli(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """write-snapshot CLI writes pending accessions minus completion log."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    snapshot = tmp_path / "snapshot.txt"
    input_file.write_text("P1\nP2\nP3\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "rockfish_queue.py",
            "write-snapshot",
            "--input",
            str(input_file),
            "--completed",
            str(completion_log),
            "-o",
            str(snapshot),
        ],
    ):
        rockfish_main()

    assert load_accession_lines(snapshot) == ["P1", "P3"]
    captured = capsys.readouterr()
    assert "Wrote snapshot" in captured.err


def test_write_snapshot_cli_excludes_failed_and_reports_counts(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """write-snapshot excludes failed IDs and prints queue counts."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    failed_log = tmp_path / "failed.txt"
    snapshot = tmp_path / "snapshot.txt"
    input_file.write_text("P1\nP2\nP3\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")
    failed_log.write_text("P3\tmissing_pdb\tno file\n", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "rockfish_queue.py",
            "write-snapshot",
            "--input",
            str(input_file),
            "--completed",
            str(completion_log),
            "--failed",
            str(failed_log),
            "-o",
            str(snapshot),
        ],
    ):
        rockfish_main()

    assert load_accession_lines(snapshot) == ["P1"]
    captured = capsys.readouterr()
    assert "failed=1" in captured.err
    assert "pending=1" in captured.err


def test_write_snapshot_cli_require_pdb_preflight(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """--require-pdb keeps only PDB-backed accessions and writes skipped log."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    snapshot = tmp_path / "snapshot.txt"
    skipped = tmp_path / "skipped.txt"
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "AF-P1-F1-model_v6.pdb").write_text("x", encoding="utf-8")
    (structures / "AF-P2-F1-model_v6.cif.gz").write_text("y", encoding="utf-8")
    input_file.write_text("P1\nP2\nP3\n", encoding="utf-8")
    completion_log.write_text("", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "rockfish_queue.py",
            "write-snapshot",
            "--input",
            str(input_file),
            "--completed",
            str(completion_log),
            "-o",
            str(snapshot),
            "--require-pdb",
            "--structures-dir",
            str(structures),
            "--skipped-output",
            str(skipped),
        ],
    ):
        rockfish_main()

    assert load_accession_lines(snapshot) == ["P1"]
    skipped_text = skipped.read_text(encoding="utf-8")
    assert "P2\tcif_only" in skipped_text
    assert "P3\tmissing_any_structure" in skipped_text
    captured = capsys.readouterr()
    assert "cif_only=1" in captured.err


def test_write_snapshot_cli_errors_when_nothing_pending(tmp_path: Path) -> None:
    """write-snapshot exits with error when every accession is already completed."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    snapshot = tmp_path / "snapshot.txt"
    input_file.write_text("P1\n", encoding="utf-8")
    completion_log.write_text("P1\n", encoding="utf-8")

    with (
        patch.object(
            sys,
            "argv",
            [
                "rockfish_queue.py",
                "write-snapshot",
                "--input",
                str(input_file),
                "--completed",
                str(completion_log),
                "-o",
                str(snapshot),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        rockfish_main()

    assert exc_info.value.code == 2


def test_prepare_cli(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """prepare-rockfish-accessions CLI writes output and reports stats."""
    fetch_json = tmp_path / "dnaj.json"
    output = tmp_path / "accessions.txt"
    fetch_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {"proteins": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P1"}}]},
                ],
            },
        ),
        encoding="utf-8",
    )

    with patch.object(
        sys,
        "argv",
        [
            "rockfish_queue.py",
            "prepare",
            str(fetch_json),
            "-o",
            str(output),
        ],
    ):
        rockfish_main()

    assert output.read_text(encoding="utf-8") == "P1\n"
    captured = capsys.readouterr()
    assert "unique_accessions=1" in captured.err
    assert "duplicate_records_skipped=1" in captured.err


def test_parse_failure_log_line_defaults_missing_reason() -> None:
    """Bare accession lines are treated as unspecified failures."""
    record = parse_failure_log_line("P1")
    assert record is not None
    assert record.accession == "P1"
    assert record.reason_code == "unspecified"
    assert record.detail == ""


def test_summarize_failures_counts_by_reason(tmp_path: Path) -> None:
    """summarize_failures aggregates reason codes in descending count order."""
    failed_log = tmp_path / "failed_pocket.txt"
    failed_log.write_text(
        "P1\tcif_only\tCIF only\nP2\tmissing_pdb\tno pdb\nP3\tcif_only\tCIF only\nP4\nP2\tmissing_pdb\tdupe row\n",
        encoding="utf-8",
    )
    summary = summarize_failures(failed_log)
    assert summary.total == 5
    assert summary.counts_by_reason == {"cif_only": 2, "missing_pdb": 2, "unspecified": 1}
    assert summary.accessions_for_reason("cif_only") == ["P1", "P3"]
    assert summary.accessions_for_reason("missing_pdb") == ["P2"]


def test_summarize_failures_cli_writes_json_and_reason_list(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """summarize-failures CLI prints counts and can export one reason cohort."""
    failed_log = tmp_path / "failed_pocket.txt"
    json_out = tmp_path / "summary.json"
    accessions_out = tmp_path / "cif_only.txt"
    failed_log.write_text(
        "P1\tcif_only\tCIF only\nP2\tanalyze_nonzero\tboom\nP3\tcif_only\tCIF only\n",
        encoding="utf-8",
    )

    with patch.object(
        sys,
        "argv",
        [
            "rockfish_queue.py",
            "summarize-failures",
            str(failed_log),
            "--json-output",
            str(json_out),
            "--reason",
            "cif_only",
            "--write-accessions",
            str(accessions_out),
        ],
    ):
        rockfish_main()

    captured = capsys.readouterr()
    assert "Total failure rows: 3" in captured.err
    assert "cif_only\t2" in captured.err
    assert load_accession_lines(accessions_out) == ["P1", "P3"]
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["counts_by_reason"]["cif_only"] == 2
