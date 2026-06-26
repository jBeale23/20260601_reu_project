"""Tests for scripts/merge_all_features.py."""

from __future__ import annotations

import csv
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.capture import CaptureFixture

from jdp_classifier.classify import JDP_DATA_COLUMNS, JDP_SOURCE_COLUMNS
from jdp_classifier.classify import OUTPUT_COLUMNS as JDP_OUTPUT_COLUMNS
from scripts.merge_all_features import (
    OUTPUT_COLUMNS,
    MergeAllInputs,
    load_jdp_table,
    merge_all_features,
    write_merged_all_csv,
)
from scripts.merge_all_features import main as merge_all_main
from scripts.merge_features import POCKET_DATA_COLUMNS, iter_fetch_records


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


def _jdp_row(
    accession: str,
    *,
    predicted_class: str = "A",
    quality_flags: str = "",
    layout_tags: str = "",
) -> dict[str, str]:
    return {
        "accession": accession,
        "protein_name": f"Protein {accession}",
        "protein_length": "400",
        "source_database": "reviewed",
        "architecture_ida": "PF00226:IPR001623-PF01556:IPR001623",
        "architecture_ida_id": "hash-a",
        "appears_in_architecture_count": "1",
        "n_domains": "3",
        "j_domain_position": "n_terminal",
        "has_gf_rich": "false",
        "has_transmembrane": "false",
        "has_signal_peptide": "false",
        "localization_source": "missing",
        "has_hpd": "true",
        "hpd_source": "interpro",
        "hpd_confidence": "high",
        "predicted_class": predicted_class,
        "class_confidence": "high",
        "layout_tags": layout_tags,
        "quality_flags": quality_flags,
    }


def _write_jdp_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JDP_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _jdp_indexes(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    jdp_by_accession = {
        row["accession"]: {
            output_column: row.get(source_column, "")
            for source_column, output_column in zip(JDP_SOURCE_COLUMNS, JDP_DATA_COLUMNS, strict=True)
        }
        for row in rows
    }
    identity_by_accession = {
        row["accession"]: {
            "protein_name": row.get("protein_name", ""),
            "protein_length": row.get("protein_length", ""),
            "source_database": row.get("source_database", ""),
        }
        for row in rows
    }
    return jdp_by_accession, identity_by_accession


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
                    "name": "No pocket data",
                    "length": 600,
                    "source_database": "unreviewed",
                },
            },
        ],
    }


def _dnaj_fetch() -> dict:
    return {
        "architectures": [
            {
                "ida": "PF00226:IPR001623-PF01556:IPR001623",
                "ida_id": "hash-j1",
                "proteins": [
                    {
                        "metadata": {
                            "accession": "P0ACJ8",
                            "name": "Chaperone protein DnaJ",
                            "length": 376,
                            "source_database": "reviewed",
                        },
                        "appears_in_architecture_count": 1,
                    },
                ],
            },
        ],
    }


def _pocket_index(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        accession: {column: row.get(column, "") for column in POCKET_DATA_COLUMNS} for accession, row in rows.items()
    }


def test_outer_join_all_sources() -> None:
    """Outer join unions accessions across DnaK, DnaJ, pocket, and JDP inputs."""
    dnak_records = iter_fetch_records(_dnak_fetch())
    dnaj_records = iter_fetch_records(_dnaj_fetch())
    pocket_by_accession = _pocket_index(
        {
            "P0A6Y8": _pocket_row("P0A6Y8"),
            "P08113": _pocket_row("P08113", quality_flags="charge_inversion_candidate"),
        },
    )
    jdp_by_accession, jdp_identity_by_accession = _jdp_indexes(
        [_jdp_row("P0ACJ8", predicted_class="A")],
    )

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={record.accession: record for record in dnak_records},
            dnaj_by_accession={record.accession: record for record in dnaj_records},
            pocket_by_accession=pocket_by_accession,
            jdp_by_accession=jdp_by_accession,
            jdp_identity_by_accession=jdp_identity_by_accession,
            provided_sources={"dnak": True, "dnaj": True, "pocket": True, "jdp": True},
            join="outer",
        ),
    )

    by_accession = {row["accession"]: row for row in merged}
    assert len(merged) == 4
    assert by_accession["P0A6Y8"]["fetch_sources"] == "dnak"
    assert by_accession["P0A6Y8"]["has_pocket_charge"] == "true"
    assert by_accession["P0A6Y8"]["has_jdp_classification"] == "false"
    assert by_accession["P0ACJ8"]["fetch_sources"] == "dnaj"
    assert by_accession["P0ACJ8"]["jdp_predicted_class"] == "A"
    assert by_accession["P0ACJ8"]["dnaj_architecture_ida"] == "PF00226:IPR001623-PF01556:IPR001623"
    assert by_accession["P08113"]["charge_inversion_candidate"] == "true"
    assert by_accession["P99999"]["has_pocket_charge"] == "false"


def test_inner_join_requires_all_provided_sources() -> None:
    """Inner join keeps only accessions present in every provided input."""
    dnak_records = iter_fetch_records(_dnak_fetch())
    pocket_by_accession = _pocket_index(
        {
            "P0A6Y8": _pocket_row("P0A6Y8"),
            "P08113": _pocket_row("P08113"),
        },
    )

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={record.accession: record for record in dnak_records},
            dnaj_by_accession={},
            pocket_by_accession=pocket_by_accession,
            jdp_by_accession={},
            jdp_identity_by_accession={},
            provided_sources={"dnak": True, "dnaj": False, "pocket": True, "jdp": False},
            join="inner",
        ),
    )

    assert {row["accession"] for row in merged} == {"P0A6Y8", "P08113"}


def test_pocket_quality_flags_renamed() -> None:
    """Unified output renames pocket quality_flags to pocket_quality_flags."""
    pocket_by_accession = _pocket_index(
        {"P0A6Y8": _pocket_row("P0A6Y8", quality_flags="charge_inversion_candidate")},
    )

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={},
            dnaj_by_accession={},
            pocket_by_accession=pocket_by_accession,
            jdp_by_accession={},
            jdp_identity_by_accession={},
            provided_sources={"dnak": False, "dnaj": False, "pocket": True, "jdp": False},
            join="outer",
        ),
    )

    assert merged[0]["pocket_quality_flags"] == "charge_inversion_candidate"
    assert "quality_flags" not in merged[0]


def test_load_jdp_table(tmp_path: Path) -> None:
    """load_jdp_table indexes jdp_-prefixed metrics and identity fields."""
    jdp_csv = tmp_path / "jdp.csv"
    _write_jdp_csv(jdp_csv, [_jdp_row("P0ACJ8")])

    jdp_by_accession, identity_by_accession = load_jdp_table(jdp_csv)

    assert jdp_by_accession["P0ACJ8"]["jdp_predicted_class"] == "A"
    assert identity_by_accession["P0ACJ8"]["protein_name"] == "Protein P0ACJ8"


def test_write_merged_all_csv_roundtrip(tmp_path: Path) -> None:
    """write_merged_all_csv emits OUTPUT_COLUMNS in stable order."""
    pocket_by_accession = _pocket_index({"P0A6Y8": _pocket_row("P0A6Y8")})
    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={},
            dnaj_by_accession={},
            pocket_by_accession=pocket_by_accession,
            jdp_by_accession={},
            jdp_identity_by_accession={},
            provided_sources={"dnak": False, "dnaj": False, "pocket": True, "jdp": False},
            join="outer",
        ),
    )

    output_csv = tmp_path / "all_features.csv"
    write_merged_all_csv(merged, output_csv)

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert rows[0]["accession"] == "P0A6Y8"


def test_merge_all_cli_smoke(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """CLI writes unified CSV from DnaK fetch and pocket CSV."""
    dnak_json = tmp_path / "dnak.json"
    pocket_csv = tmp_path / "pocket.csv"
    output_csv = tmp_path / "all_features.csv"

    dnak_json.write_text(json.dumps(_dnak_fetch()))
    _write_pocket_csv(pocket_csv, [_pocket_row("P0A6Y8"), _pocket_row("P08113")])

    with patch.object(
        sys,
        "argv",
        [
            "merge_all_features.py",
            "--dnak-json",
            str(dnak_json),
            "--pocket-csv",
            str(pocket_csv),
            "-o",
            str(output_csv),
        ],
    ):
        merge_all_main()

    captured = capsys.readouterr()
    assert "Merged 3 row(s)" in captured.err

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[0]["fetch_sources"] == "dnak"
    assert "jdp_predicted_class" in rows[0]
