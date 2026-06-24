"""Tests for scripts/extract_uniprot_ids.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_uniprot_ids import extract_accessions, main


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
