"""Tests for scripts/extract_uniprot_ids.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_uniprot_ids import extract_accessions, fetch_warnings, main


def test_extract_accessions_from_dnak_output() -> None:
    """Extract accessions from a DnaK-style JSON file."""
    data = {
        "proteins": [
            {"metadata": {"accession": "P0A6Y8"}},
            {"metadata": {"accession": "P12345"}},
            {"metadata": {}},
        ],
    }
    assert extract_accessions(data) == ["P0A6Y8", "P12345"]


def test_extract_accessions_from_dnaj_output() -> None:
    """Extract unique accessions from a DnaJ-style JSON file."""
    data = {
        "architectures": [
            {
                "proteins": [
                    {"metadata": {"accession": "P1"}},
                    {"metadata": {"accession": "P2"}},
                ],
            },
            {
                "proteins": [
                    {"metadata": {"accession": "P2"}},
                    {"metadata": {"accession": "P3"}},
                ],
            },
        ],
    }
    assert extract_accessions(data) == ["P1", "P2", "P3"]


def test_extract_accessions_cli(tmp_path: Path) -> None:
    """CLI writes one accession per line."""
    json_file = tmp_path / "proteins.json"
    json_file.write_text(json.dumps({"proteins": [{"metadata": {"accession": "P9"}}]}))
    out_file = tmp_path / "ids.txt"

    with patch.object(sys, "argv", ["extract_uniprot_ids.py", str(json_file), "-o", str(out_file)]):
        main()

    assert out_file.read_text() == "P9\n"


def test_extract_accessions_empty_input() -> None:
    """Empty fetch output yields no accessions."""
    assert extract_accessions({}) == []


def test_extract_accessions_cli_missing_file(tmp_path: Path) -> None:
    """CLI exits with an error when the JSON file does not exist."""
    missing = tmp_path / "missing.json"

    with (
        patch.object(sys, "argv", ["extract_uniprot_ids.py", str(missing)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 2


def test_extract_accessions_cli_invalid_json(tmp_path: Path) -> None:
    """CLI exits with an error when the JSON file is malformed."""
    json_file = tmp_path / "bad.json"
    json_file.write_text("{not valid json")

    with (
        patch.object(sys, "argv", ["extract_uniprot_ids.py", str(json_file)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 2


def test_fetch_warnings_dnak_partial() -> None:
    """Flag incomplete DnaK fetch output."""
    warnings = fetch_warnings({"is_partial": True, "proteins": []})
    assert len(warnings) == 1
    assert "is_partial" in warnings[0]


def test_fetch_warnings_dnaj_partial_architectures() -> None:
    """Flag incomplete DnaJ architecture fetch output."""
    warnings = fetch_warnings(
        {
            "architectures": [
                {"is_partial": False, "proteins": []},
                {"is_partial": True, "proteins": []},
            ],
        },
    )
    assert len(warnings) == 1
    assert "1 architecture" in warnings[0]


def test_extract_accessions_cli_warns_on_empty_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI warns when no accessions are extracted."""
    json_file = tmp_path / "empty.json"
    json_file.write_text("{}")

    with patch.object(sys, "argv", ["extract_uniprot_ids.py", str(json_file)]):
        main()

    captured = capsys.readouterr()
    assert "WARNING: No UniProt accessions found" in captured.err


def test_extract_accessions_cli_warns_on_partial_dnak(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI warns when source JSON is a partial DnaK fetch."""
    json_file = tmp_path / "partial.json"
    json_file.write_text(json.dumps({"is_partial": True, "proteins": []}))

    with patch.object(sys, "argv", ["extract_uniprot_ids.py", str(json_file)]):
        main()

    captured = capsys.readouterr()
    assert "WARNING: DnaK fetch output is marked is_partial=true" in captured.err
