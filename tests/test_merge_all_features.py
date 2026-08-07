"""Tests for scripts/merge_all_features.py."""

from __future__ import annotations

import csv
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.capture import CaptureFixture

from domain_layout.constants import FEATURE_COLUMNS
from jdp_classifier.classify import JDP_DATA_COLUMNS, JDP_SOURCE_COLUMNS
from jdp_classifier.classify import OUTPUT_COLUMNS as JDP_OUTPUT_COLUMNS
from scripts.merge_all_features import (
    DOMAIN_DATA_COLUMNS,
    OUTPUT_COLUMNS,
    MergeAllInputs,
    load_domain_layout_table,
    load_jdp_table,
    merge_all_features,
    write_merged_all_csv,
)
from scripts.merge_all_features import main as merge_all_main
from scripts.merge_features import POCKET_DATA_COLUMNS, FetchRecord, iter_fetch_records


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
            motif_by_accession={},
            provided_sources={"dnak": True, "dnaj": True, "pocket": True, "jdp": True, "motif": False},
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
    assert by_accession["P0ACJ8"]["chaperone_system_membership"] == "dnaj"
    assert by_accession["P0A6Y8"]["chaperone_system_membership"] == "dnak"
    assert by_accession["P0ACJ8"]["unified_confidence_tier"] == "high"
    assert "jdp_class_a" in by_accession["P0ACJ8"]["classification_tags"]


def test_dual_homolog_gets_dual_membership_and_unified_tier() -> None:
    """Accessions in both fetches are labeled dual with combined confidence."""
    dnak_records = iter_fetch_records(_dnak_fetch())
    dnaj_records = iter_fetch_records(_dnaj_fetch())
    pocket_by_accession = _pocket_index({"P0ACJ8": _pocket_row("P0ACJ8", mapping_confidence="medium")})
    jdp_by_accession, jdp_identity_by_accession = _jdp_indexes([_jdp_row("P0ACJ8", predicted_class="A")])
    template = dnak_records[0]
    dual_dnak_record = FetchRecord(
        accession="P0ACJ8",
        protein_name=template.protein_name,
        protein_length=template.protein_length,
        source_database=template.source_database,
        fetch_source="dnak",
        architecture_ida="",
        architecture_ida_id="",
        appears_in_architecture_count="",
    )

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={record.accession: record for record in dnak_records} | {"P0ACJ8": dual_dnak_record},
            dnaj_by_accession={record.accession: record for record in dnaj_records},
            pocket_by_accession=pocket_by_accession,
            jdp_by_accession=jdp_by_accession,
            jdp_identity_by_accession=jdp_identity_by_accession,
            motif_by_accession={},
            provided_sources={"dnak": True, "dnaj": True, "pocket": True, "jdp": True, "motif": False},
            join="outer",
        ),
    )
    by_accession = {row["accession"]: row for row in merged}
    assert by_accession["P0ACJ8"]["chaperone_system_membership"] == "dual"
    assert by_accession["P0ACJ8"]["unified_confidence_tier"] == "medium"
    assert "dual_chaperone_homolog" in by_accession["P0ACJ8"]["classification_tags"]


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
            motif_by_accession={},
            provided_sources={"dnak": True, "dnaj": False, "pocket": True, "jdp": False, "motif": False},
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
            motif_by_accession={},
            provided_sources={"dnak": False, "dnaj": False, "pocket": True, "jdp": False, "motif": False},
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
            motif_by_accession={},
            provided_sources={"dnak": False, "dnaj": False, "pocket": True, "jdp": False, "motif": False},
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


def _domain_layout_row(
    accession: str,
    *,
    predicted_class: str = "A",
    predicted_subclass: str = "a_canonical",
    novel: str = "false",
    novelty_score: str = "0.0000",
) -> dict[str, str]:
    return {
        "accession": accession,
        "protein_name": f"Protein {accession}",
        "protein_length": "376",
        "organism_name": "Escherichia coli",
        "n_entries": "5",
        "n_structured_domains": "4",
        "domain_family_layout": "j_domain>dnaj_c>zinc_finger_like",
        "j_domain_position": "n_terminal",
        "has_j_domain": "true",
        "has_hpd": "true",
        "has_dnaj_c": "true",
        "has_zinc_finger_like": "true",
        "has_gf_rich_region": "true",
        "has_transmembrane": "false",
        "has_signal_peptide": "false",
        "idr_fraction": "0.0426",
        "mean_disorder": "0.5000",
        "disorder_backend": "foldindex",
        "n_msa_regions": "5",
        "n_shark_regions": "1",
        "msa_residues": "315",
        "shark_residues": "40",
        "shark_backend": "blosum_kmer",
        "shark_best_reference": "P08622|77-116|linker",
        "shark_best_reference_class": "A",
        "shark_best_similarity": "0.9000",
        "layout_predicted_class": predicted_class,
        "layout_predicted_subclass": predicted_subclass,
        "layout_class_confidence": "high",
        "layout_novelty_score": novelty_score,
        "novel_class_candidate": novel,
        "layout_evidence_tags": "novel_class_candidate" if novel == "true" else "",
    }


def _write_domain_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_load_domain_layout_table_prefixes_columns(tmp_path: Path) -> None:
    """Layout columns are namespaced so they never clash with the v1 classifier."""
    path = tmp_path / "domain_layout_features.csv"
    _write_domain_csv(path, [_domain_layout_row("P08622")])

    table = load_domain_layout_table(path)
    assert set(table["P08622"]) == set(DOMAIN_DATA_COLUMNS)
    assert table["P08622"]["domain_layout_predicted_class"] == "A"
    assert table["P08622"]["domain_shark_backend"] == "blosum_kmer"


def test_load_domain_layout_table_rejects_missing_columns(tmp_path: Path) -> None:
    """A truncated layout CSV is rejected with the missing column names."""
    path = tmp_path / "bad.csv"
    path.write_text("accession,layout_predicted_class\nP08622,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_domain_layout_table(path)


def test_merge_includes_domain_layout_and_novel_flag(tmp_path: Path) -> None:
    """Layout rows flow into the unified table with a top-level novel-class flag."""
    domain_csv = tmp_path / "domain_layout_features.csv"
    _write_domain_csv(
        domain_csv,
        [
            _domain_layout_row("P08622"),
            _domain_layout_row(
                "P99999",
                predicted_class="C",
                predicted_subclass="c_atypical_multi_domain",
                novel="true",
                novelty_score="0.7500",
            ),
        ],
    )

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={},
            dnaj_by_accession={},
            pocket_by_accession={},
            jdp_by_accession={},
            jdp_identity_by_accession={},
            motif_by_accession={},
            domain_by_accession=load_domain_layout_table(domain_csv),
            provided_sources={"domain": True},
        ),
    )

    by_accession = {row["accession"]: row for row in merged}
    assert by_accession["P08622"]["has_domain_layout"] == "true"
    assert by_accession["P08622"]["novel_class_candidate"] == "false"
    assert by_accession["P99999"]["novel_class_candidate"] == "true"
    assert "novel_class_candidate" in by_accession["P99999"]["classification_tags"]
    assert "layout_c_atypical_multi_domain" in by_accession["P99999"]["classification_tags"]
    assert by_accession["P08622"]["unified_confidence_tier"] == "high"


def test_merge_leaves_domain_columns_empty_without_layout_input() -> None:
    """Accessions without layout data keep empty layout columns and a false flag."""
    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={},
            dnaj_by_accession={},
            pocket_by_accession={},
            jdp_by_accession={"P1": dict.fromkeys(JDP_DATA_COLUMNS, "")},
            jdp_identity_by_accession={},
            motif_by_accession={},
            provided_sources={"jdp": True},
        ),
    )
    assert merged[0]["has_domain_layout"] == "false"
    assert merged[0]["novel_class_candidate"] == "false"
    assert all(merged[0][column] == "" for column in DOMAIN_DATA_COLUMNS)


def test_inner_join_requires_domain_layout_when_provided(tmp_path: Path) -> None:
    """--join inner drops accessions missing from the layout table."""
    domain_csv = tmp_path / "domain_layout_features.csv"
    _write_domain_csv(domain_csv, [_domain_layout_row("P08622")])

    merged = merge_all_features(
        MergeAllInputs(
            dnak_by_accession={},
            dnaj_by_accession={},
            pocket_by_accession={},
            jdp_by_accession={"P08622": dict.fromkeys(JDP_DATA_COLUMNS, ""), "P2": dict.fromkeys(JDP_DATA_COLUMNS, "")},
            jdp_identity_by_accession={},
            motif_by_accession={},
            domain_by_accession=load_domain_layout_table(domain_csv),
            provided_sources={"jdp": True, "domain": True},
            join="inner",
        ),
    )
    assert [row["accession"] for row in merged] == ["P08622"]


def test_merge_all_cli_with_domain_csv(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI accepts --domain-csv on its own and reports novel candidates."""
    domain_csv = tmp_path / "domain_layout_features.csv"
    _write_domain_csv(
        domain_csv,
        [_domain_layout_row("P99999", novel="true", predicted_subclass="n_novel_candidate")],
    )
    output = tmp_path / "all_features.csv"

    with patch.object(sys, "argv", ["merge-all-features", "--domain-csv", str(domain_csv), "-o", str(output)]):
        merge_all_main()

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["domain_layout_predicted_subclass"] == "n_novel_candidate"
    assert "1 novel-class candidate(s)" in capsys.readouterr().err
