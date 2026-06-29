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
    extraction_stats,
    load_accession_lines,
    pending_accessions,
    prepare_from_fetch_json,
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


def test_pending_accessions_excludes_completed(tmp_path: Path) -> None:
    """Pending queue excludes accessions already in the completion log."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    input_file.write_text("P1\nP2\nP3\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")
    assert pending_accessions(input_file, completion_log) == ["P1", "P3"]


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
