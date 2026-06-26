"""Tests for scripts/merge_features.py."""

from __future__ import annotations

import csv
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.capture import CaptureFixture

from scripts.merge_features import (
    POCKET_DATA_COLUMNS,
    iter_fetch_records,
    load_pocket_table,
    merge_records,
    write_merged_csv,
)
from scripts.merge_features import main as merge_main


def _pocket_row(
    accession: str,
    *,
    quality_flags: str = "",
    mapping_confidence: str = "high",
) -> dict[str, str]:
    return {
        "accession": accession,
        "confidence_tier": "high",
        "mapping_confidence": mapping_confidence,
        "conservation_score": "high",
        "mapping_mode": "sbd_local",
        "contact_net_charge": "0",
        "shell_net_charge": "0",
        "delta_contact_net_charge": "0",
        "delta_net_charge_vs_reference": "0",
        "contact_mapping_fraction": "1.0",
        "contact_sequence_identity": "1.0",
        "mean_contact_ca_distance": "0.0",
        "contact_drmsd": "0.0",
        "n_contacts_within_5A": "17",
        "sbd_alignment_coverage": "1.0",
        "mean_plddt_pocket": "90.0",
        "mapped_pocket_residues": "50",
        "n_histidine": "1",
        "net_charge_excluding_his": "-1",
        "exposed_net_charge": "0",
        "charge_density": "-0.02",
        "quality_flags": quality_flags,
    }


def _write_pocket_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(_pocket_row("X").keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dnak_fetch() -> dict:
    return {
        "proteins": [
            {
                "metadata": {
                    "accession": "P0A6Y8",
                    "name": "Chaperone protein DnaK",
                    "length": 638,
                    "source_database": "reviewed",
                },
            },
            {
                "metadata": {
                    "accession": "P08113",
                    "name": "HSP70",
                    "length": 642,
                    "source_database": "reviewed",
                },
            },
            {
                "metadata": {
                    "accession": "P99999",
                    "name": "No structure yet",
                    "length": 600,
                    "source_database": "unreviewed",
                },
            },
        ],
    }


def _pocket_index(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        accession: {column: row.get(column, "") for column in POCKET_DATA_COLUMNS} for accession, row in rows.items()
    }


def test_dnak_left_join() -> None:
    """Three fetch proteins and two pocket rows yield three merged rows."""
    fetch_records = iter_fetch_records(_dnak_fetch())
    pocket_by_accession = _pocket_index(
        {
            "P0A6Y8": _pocket_row("P0A6Y8"),
            "P08113": _pocket_row("P08113"),
        },
    )
    merged, orphan_count = merge_records(fetch_records, pocket_by_accession, join="left")

    assert len(merged) == 3
    assert orphan_count == 0
    by_accession = {row["accession"]: row for row in merged}
    assert by_accession["P0A6Y8"]["has_pocket_charge"] == "true"
    assert by_accession["P08113"]["has_pocket_charge"] == "true"
    assert by_accession["P99999"]["has_pocket_charge"] == "false"
    assert by_accession["P99999"]["mapping_confidence"] == ""


def test_dnak_inner_join() -> None:
    """Inner join keeps only fetch proteins with pocket metrics."""
    fetch_records = iter_fetch_records(_dnak_fetch())
    pocket_by_accession = _pocket_index(
        {
            "P0A6Y8": _pocket_row("P0A6Y8"),
            "P08113": _pocket_row("P08113"),
        },
    )

    merged, _ = merge_records(fetch_records, pocket_by_accession, join="inner")

    assert len(merged) == 2
    assert {row["accession"] for row in merged} == {"P0A6Y8", "P08113"}


def test_merge_join_normalizes_fetch_accession_whitespace() -> None:
    """Fetch metadata whitespace is stripped so pocket rows join correctly."""
    data = {
        "proteins": [
            {"metadata": {"accession": " P0A6Y8 ", "name": "DnaK"}},
        ],
    }
    fetch_records = iter_fetch_records(data)
    pocket_by_accession = _pocket_index({"P0A6Y8": _pocket_row("P0A6Y8")})

    merged, _ = merge_records(fetch_records, pocket_by_accession, join="left")

    assert merged[0]["accession"] == "P0A6Y8"
    assert merged[0]["has_pocket_charge"] == "true"


def test_dnaj_dedupe() -> None:
    """Deduplicated DnaJ rows aggregate architecture IDs per accession."""
    data = {
        "architectures": [
            {
                "ida": "PF00226:IPR001623",
                "ida_id": "hash-a",
                "proteins": [
                    {
                        "metadata": {"accession": "P1", "name": "Jdp1"},
                        "appears_in_architecture_count": 2,
                    },
                ],
            },
            {
                "ida": "PF00564:IPR001623",
                "ida_id": "hash-b",
                "proteins": [
                    {
                        "metadata": {"accession": "P1", "name": "Jdp1"},
                        "appears_in_architecture_count": 2,
                    },
                ],
            },
        ],
    }

    records = iter_fetch_records(data, dnaj_rows="dedupe")

    assert len(records) == 1
    assert records[0].accession == "P1"
    assert "PF00226:IPR001623" in records[0].architecture_ida
    assert "PF00564:IPR001623" in records[0].architecture_ida
    assert "hash-a" in records[0].architecture_ida_id
    assert "hash-b" in records[0].architecture_ida_id


def test_dnaj_explode() -> None:
    """Exploded DnaJ rows keep one row per architecture membership."""
    data = {
        "architectures": [
            {
                "ida": "PF00226:IPR001623",
                "ida_id": "hash-a",
                "proteins": [{"metadata": {"accession": "P1"}}],
            },
            {
                "ida": "PF00564:IPR001623",
                "ida_id": "hash-b",
                "proteins": [{"metadata": {"accession": "P1"}}],
            },
        ],
    }

    records = iter_fetch_records(data, dnaj_rows="explode")

    assert len(records) == 2
    assert {record.architecture_ida for record in records} == {
        "PF00226:IPR001623",
        "PF00564:IPR001623",
    }


def test_charge_inversion_candidate() -> None:
    """quality_flags containing charge_inversion_candidate sets derived boolean."""
    fetch_records = iter_fetch_records(_dnak_fetch())
    pocket_by_accession = _pocket_index(
        {
            "P0A6Y8": _pocket_row("P0A6Y8"),
            "P08113": _pocket_row("P08113", quality_flags="charge_inversion_candidate"),
        },
    )

    merged, _ = merge_records(fetch_records, pocket_by_accession, join="inner")
    by_accession = {row["accession"]: row for row in merged}

    assert by_accession["P08113"]["charge_inversion_candidate"] == "true"
    assert by_accession["P0A6Y8"]["charge_inversion_candidate"] == "false"


def test_orphan_pocket_warning_count(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Pocket CSV accessions missing from fetch are counted on stderr."""
    fetch_json = tmp_path / "fetch.json"
    pocket_csv = tmp_path / "pocket.csv"
    output_csv = tmp_path / "merged.csv"

    fetch_json.write_text(json.dumps(_dnak_fetch()))
    _write_pocket_csv(
        pocket_csv,
        [
            _pocket_row("P0A6Y8"),
            _pocket_row("ORPHAN1"),
        ],
    )

    with patch.object(
        sys,
        "argv",
        [
            "merge_features.py",
            str(fetch_json),
            "--pocket-csv",
            str(pocket_csv),
            "-o",
            str(output_csv),
        ],
    ):
        merge_main()

    captured = capsys.readouterr()
    assert "1 orphan pocket CSV accessions" in captured.err


def test_merge_cli_smoke(tmp_path: Path) -> None:
    """CLI writes merged CSV with expected header and row count."""
    fetch_json = tmp_path / "fetch.json"
    pocket_csv = tmp_path / "pocket.csv"
    output_csv = tmp_path / "merged.csv"

    fetch_json.write_text(json.dumps(_dnak_fetch()))
    _write_pocket_csv(pocket_csv, [_pocket_row("P0A6Y8"), _pocket_row("P08113")])

    with patch.object(
        sys,
        "argv",
        [
            "merge_features.py",
            str(fetch_json),
            "--pocket-csv",
            str(pocket_csv),
            "-o",
            str(output_csv),
        ],
    ):
        merge_main()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[0]["accession"] == "P0A6Y8"
    assert "mapping_confidence" in rows[0]


def test_load_pocket_table(tmp_path: Path) -> None:
    """load_pocket_table indexes rows by accession."""
    pocket_csv = tmp_path / "pocket.csv"
    _write_pocket_csv(pocket_csv, [_pocket_row("P0A6Y8")])

    pocket_by_accession = load_pocket_table(pocket_csv)

    assert set(pocket_by_accession) == {"P0A6Y8"}
    assert pocket_by_accession["P0A6Y8"]["mapping_confidence"] == "high"


def test_write_merged_csv_roundtrip(tmp_path: Path) -> None:
    """write_merged_csv emits OUTPUT_COLUMNS in stable order."""
    fetch_records = iter_fetch_records(_dnak_fetch())
    pocket_by_accession = _pocket_index({"P0A6Y8": _pocket_row("P0A6Y8")})
    merged, _ = merge_records(fetch_records, pocket_by_accession, join="left")

    output_csv = tmp_path / "merged.csv"
    write_merged_csv(merged, output_csv)

    text = output_csv.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header.startswith("accession,protein_name,protein_length,source_database,fetch_source")
