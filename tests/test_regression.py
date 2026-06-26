"""Cross-module regression tests for the full analysis pipeline."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from jdp_classifier.classify import (
    JDP_DATA_COLUMNS,
    JDP_SOURCE_COLUMNS,
    classify_fetch_json,
    write_classifications_csv,
)
from jdp_classifier.classify import (
    OUTPUT_COLUMNS as JDP_OUTPUT_COLUMNS,
)
from scripts.extract_uniprot_ids import extract_accessions
from scripts.merge_all_features import OUTPUT_COLUMNS as MERGE_ALL_OUTPUT_COLUMNS
from scripts.merge_all_features import main as merge_all_main
from scripts.merge_features import POCKET_DATA_COLUMNS, iter_fetch_records, merge_records
from scripts.rockfish_queue import (
    load_accession_lines,
    pending_accessions,
    prepare_from_fetch_json,
    write_array_snapshot,
)
from structure_analysis.analyze import (
    CSV_COLUMNS,
    BatchAnalysisOptions,
    PocketChargeResult,
    _per_accession_result_json_paths,
    analyze_directory,
    merge_results_directory,
    result_from_dict,
    write_result,
)
from structure_analysis.geometry import apply_transform, kabsch_superpose
from structure_analysis.pdb_io import load_structure_residues


def test_pocket_merge_columns_match_structure_csv_export() -> None:
    """merge-features pocket columns stay aligned with analyze-pocket-charge CSV output."""
    assert [column for column in CSV_COLUMNS if column != "accession"] == POCKET_DATA_COLUMNS


def test_jdp_source_columns_align_with_classifier_output() -> None:
    """merge-all-features JDP column mapping matches classify-jdp CSV headers."""
    assert len(JDP_SOURCE_COLUMNS) == len(JDP_DATA_COLUMNS)
    for source_column in JDP_SOURCE_COLUMNS:
        assert source_column in JDP_OUTPUT_COLUMNS


def test_merge_all_output_includes_all_jdp_data_columns() -> None:
    """Unified CSV exposes every jdp_-prefixed metric column."""
    for column in JDP_DATA_COLUMNS:
        assert column in MERGE_ALL_OUTPUT_COLUMNS


def test_rockfish_prepare_matches_extract_accessions(tmp_path: Path) -> None:
    """prepare-rockfish-accessions and extract-uniprot-ids yield the same unique ID set."""
    fetch_json = tmp_path / "dnaj.json"
    data = {
        "total_proteins_fetched": 3,
        "architectures": [
            {
                "proteins": [
                    {"metadata": {"accession": "P1"}},
                    {"metadata": {"accession": "P2"}},
                ],
            },
            {"proteins": [{"metadata": {"accession": "P2"}}, {"metadata": {"accession": " P3 "}}]},
        ],
    }
    fetch_json.write_text(json.dumps(data), encoding="utf-8")

    prepared, _stats = prepare_from_fetch_json(fetch_json, wk_dir=tmp_path / "wk")
    extracted = extract_accessions(data)

    assert prepared == extracted
    assert prepared == ["P1", "P2", "P3"]


def test_array_snapshot_lines_are_unique(tmp_path: Path) -> None:
    """Snapshot rows must not repeat accessions (one task ID per protein)."""
    snapshot_path = tmp_path / "snapshot.txt"
    pending = ["A", "B", "C"]
    write_array_snapshot(pending, snapshot_path, limit=10)
    lines = load_accession_lines(snapshot_path)
    assert len(lines) == len(set(lines))


def test_pending_queue_excludes_completed_preserves_order(tmp_path: Path) -> None:
    """Rockfish pending list is stable and excludes finished accessions."""
    input_file = tmp_path / "incomplete_accessions.txt"
    completion_log = tmp_path / "completed.txt"
    input_file.write_text("P1\nP2\nP3\nP2\n", encoding="utf-8")
    completion_log.write_text("P2\n", encoding="utf-8")

    assert pending_accessions(input_file, completion_log) == ["P1", "P3"]


def test_kabsch_superpose_zero_rmsd_for_identical_coordinates() -> None:
    """Identical point sets superpose with near-zero RMSD (covariance orientation regression)."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    rotation, translation = kabsch_superpose(coords, coords.copy())
    aligned = apply_transform(coords, rotation, translation)
    rmsd = float(np.sqrt(np.mean(np.sum((coords - aligned) ** 2, axis=1))))
    assert rmsd < 1e-10


@pytest.mark.parametrize("angle", [0.0, np.pi / 6, np.pi / 3, np.pi / 2])
def test_kabsch_superpose_recovers_known_transform(angle: float) -> None:
    """Kabsch recovers rotation+translation for varied orientations (geometry regression)."""
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    rotation_true = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    translation_true = np.array([2.0, -1.0, 0.5])
    tgt = (rotation_true @ ref.T).T + translation_true

    rotation, translation = kabsch_superpose(ref, tgt)
    aligned = apply_transform(tgt, rotation, translation)
    assert np.allclose(aligned, ref, atol=1e-5)


def test_pocket_charge_result_roundtrip_preserves_fields() -> None:
    """JSON serialization roundtrip keeps fields used by merge-features."""
    original = PocketChargeResult(
        accession="PTEST",
        structure_path="AF-PTEST-F1-model_v6.pdb",
        pocket_definition="dnak_sbd_v1",
        reference_accession="P0A6Y8",
        contact_residues_expected=17,
        contact_residues_mapped=17,
        contact_mapping_fraction=1.0,
        contact_sequence_identity=0.5,
        n_contact_mismatches=0,
        mapping_mode="sbd_local",
        sbd_alignment_coverage=1.0,
        contact_net_charge=0,
        shell_net_charge=-1,
        n_positive=1,
        n_negative=2,
        n_histidine=1,
        net_charge=-1,
        net_charge_excluding_his=-2,
        exposed_net_charge=0,
        n_buried_charged=0,
        n_hydrophobic=10,
        mapped_pocket_residues=40,
        charge_density=-0.02,
        mean_plddt_pocket=85.0,
        n_low_plddt_excluded=0,
        delta_contact_net_charge=1,
        delta_net_charge_vs_reference=2,
        mean_contact_ca_distance=1.5,
        max_contact_ca_distance=3.0,
        n_contacts_within_5A=15,
        sbd_superposition_rmsd=2.0,
        contact_drmsd=1.0,
        mapping_confidence="high",
        conservation_score="medium",
        confidence_tier="medium",
        quality_flags=["charge_inversion_candidate"],
        contact_residue_details=[],
        warnings=[],
    )
    restored = result_from_dict(asdict(original))
    assert restored.accession == original.accession
    assert restored.mapping_confidence == original.mapping_confidence
    assert restored.delta_contact_net_charge == original.delta_contact_net_charge
    assert restored.quality_flags == original.quality_flags


def test_classify_jdp_output_feeds_merge_all_features(tmp_path: Path) -> None:
    """classify-jdp CSV columns are consumable by merge-all-features without rename drift."""
    dnaj_json = tmp_path / "dnaj.json"
    jdp_csv = tmp_path / "jdp.csv"
    dnaj_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623",
                        "ida_id": "hash-a",
                        "proteins": [
                            {
                                "metadata": {
                                    "accession": "P0ACJ8",
                                    "name": "DnaJ",
                                    "length": 376,
                                    "source_database": "reviewed",
                                    "sequence": "A" * 50 + "HPD" + "G" * 100,
                                },
                                "appears_in_architecture_count": 1,
                                "entries": [
                                    {
                                        "accession": "IPR001623",
                                        "entry_protein_locations": [
                                            {"fragments": [{"start": 51, "end": 53}]},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    rows, _fetches = classify_fetch_json(json.loads(dnaj_json.read_text(encoding="utf-8")), dnaj_rows="dedupe")
    write_classifications_csv(rows, jdp_csv)

    with jdp_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == JDP_OUTPUT_COLUMNS
        jdp_row = next(reader)

    for source_column, _data_column in zip(JDP_SOURCE_COLUMNS, JDP_DATA_COLUMNS, strict=True):
        assert jdp_row[source_column] != "" or source_column in {"layout_tags", "quality_flags"}


def test_merge_accession_keys_never_duplicate_in_left_join() -> None:
    """Left join emits exactly one row per fetch accession (no duplicate keys)."""
    data = {
        "proteins": [
            {"metadata": {"accession": "P1", "name": "A"}},
            {"metadata": {"accession": "P2", "name": "B"}},
        ],
    }
    fetch_records = iter_fetch_records(data)
    merged, _ = merge_records(fetch_records, {}, join="left")
    accessions = [row["accession"] for row in merged]
    assert accessions == sorted(accessions)
    assert len(accessions) == len(set(accessions))


def test_load_structure_residues_reads_gzipped_pdb(tmp_path: Path) -> None:
    """AlphaFold affetch .pdb.gz outputs are readable by the structure loader."""
    pdb_path = tmp_path / "mini.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      2  CA  LYS A   2       3.800   0.000   0.000  1.00 85.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    gz_path = tmp_path / "mini.pdb.gz"
    gz_path.write_bytes(gzip.compress(pdb_path.read_bytes()))

    plain_count = len(load_structure_residues(pdb_path))
    gz_count = len(load_structure_residues(gz_path))
    assert plain_count == gz_count == 2


def test_analyze_directory_discovers_gzipped_structures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """analyze_directory and sensitivity sweeps include *.pdb.gz files (not only *.pdb)."""
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    (structures_dir / "AF-P1-F1-model_v6.pdb").write_text("END\n", encoding="utf-8")
    gz_path = structures_dir / "AF-P2-F1-model_v6.pdb.gz"
    gz_path.write_bytes(gzip.compress(b"END\n"))

    discovered: list[str] = []

    def _record(path: Path, _context: object) -> PocketChargeResult:
        discovered.append(path.name)
        return result_from_dict(
            {
                "accession": path.stem.removeprefix("AF-").split("-F1")[0],
                "structure_path": str(path),
                "pocket_definition": "test",
                "reference_accession": "P0A6Y8",
                "contact_residues_expected": 0,
                "contact_residues_mapped": 0,
                "contact_mapping_fraction": 0.0,
                "contact_sequence_identity": 0.0,
                "n_contact_mismatches": 0,
                "mapping_mode": "sbd_local",
                "sbd_alignment_coverage": 0.0,
                "contact_net_charge": 0,
                "shell_net_charge": 0,
                "n_positive": 0,
                "n_negative": 0,
                "n_histidine": 0,
                "net_charge": 0,
                "net_charge_excluding_his": 0,
                "exposed_net_charge": 0,
                "n_buried_charged": 0,
                "n_hydrophobic": 0,
                "mapped_pocket_residues": 0,
                "charge_density": 0.0,
                "mean_plddt_pocket": 0.0,
                "n_low_plddt_excluded": 0,
                "delta_contact_net_charge": 0,
                "delta_net_charge_vs_reference": 0,
                "mean_contact_ca_distance": None,
                "max_contact_ca_distance": None,
                "n_contacts_within_5A": 0,
                "sbd_superposition_rmsd": None,
                "contact_drmsd": None,
                "mapping_confidence": "low",
                "conservation_score": "low",
                "confidence_tier": "low",
                "quality_flags": [],
                "contact_residue_details": [],
                "warnings": [],
            },
        )

    monkeypatch.setattr("structure_analysis.analyze.analyze_structure", _record)
    monkeypatch.setattr(
        "structure_analysis.analyze._build_reference_context",
        lambda _pocket_ref, _ref_path: object(),
    )
    monkeypatch.setattr("structure_analysis.analyze.load_pocket_reference", lambda _p: object())

    analyze_directory(
        BatchAnalysisOptions(
            structures_dir=structures_dir,
            pocket_ref_path=tmp_path / "pocket.yaml",
            reference_structure_path=tmp_path / "ref.pdb",
            output_dir=tmp_path / "out",
        ),
    )

    assert "AF-P1-F1-model_v6.pdb" in discovered
    assert "AF-P2-F1-model_v6.pdb.gz" in discovered


def test_merge_results_directory_skips_aggregate_summary_file(tmp_path: Path) -> None:
    """merge_results_directory must not load pocket_charge_summary.json as a single result."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_result(
        result_from_dict(
            {
                "accession": "P08113",
                "structure_path": "AF-P08113-F1-model_v6.pdb",
                "pocket_definition": "dnak_sbd_v1",
                "reference_accession": "P0A6Y8",
                "contact_residues_expected": 17,
                "contact_residues_mapped": 17,
                "contact_mapping_fraction": 1.0,
                "contact_sequence_identity": 1.0,
                "n_contact_mismatches": 0,
                "mapping_mode": "sbd_local",
                "sbd_alignment_coverage": 1.0,
                "contact_net_charge": 0,
                "shell_net_charge": 0,
                "n_positive": 0,
                "n_negative": 0,
                "n_histidine": 0,
                "net_charge": 0,
                "net_charge_excluding_his": 0,
                "exposed_net_charge": 0,
                "n_buried_charged": 0,
                "n_hydrophobic": 0,
                "mapped_pocket_residues": 40,
                "charge_density": 0.0,
                "mean_plddt_pocket": 85.0,
                "n_low_plddt_excluded": 0,
                "delta_contact_net_charge": 0,
                "delta_net_charge_vs_reference": 0,
                "mean_contact_ca_distance": 0.0,
                "max_contact_ca_distance": 0.0,
                "n_contacts_within_5A": 0,
                "sbd_superposition_rmsd": 0.0,
                "contact_drmsd": 0.0,
                "mapping_confidence": "high",
                "conservation_score": "high",
                "confidence_tier": "high",
                "quality_flags": [],
                "contact_residue_details": [],
                "warnings": [],
            },
        ),
        results_dir / "pocket_charge_P08113.json",
    )
    merge_results_directory(results_dir, results_dir)
    paths = _per_accession_result_json_paths(results_dir)
    assert all(path.name != "pocket_charge_summary.json" for path in paths)
    assert len(paths) == 1

    merged = merge_results_directory(results_dir, results_dir)
    assert len(merged) == 1
    assert merged[0].accession == "P08113"


def test_end_to_end_merge_all_with_classify_jdp_csv(tmp_path: Path) -> None:
    """Full table merge succeeds when JDP classifications are produced from the same fetch JSON."""
    dnak_json = tmp_path / "dnak.json"
    dnaj_json = tmp_path / "dnaj.json"
    jdp_csv = tmp_path / "jdp.csv"
    pocket_csv = tmp_path / "pocket.csv"
    output_csv = tmp_path / "all_features.csv"

    dnak_json.write_text(
        json.dumps(
            {
                "proteins": [
                    {"metadata": {"accession": "P0A6Y8", "name": "DnaK", "length": 638, "source_database": "reviewed"}},
                ],
            },
        ),
        encoding="utf-8",
    )
    dnaj_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623",
                        "ida_id": "h1",
                        "proteins": [
                            {
                                "metadata": {
                                    "accession": "P0ACJ8",
                                    "name": "DnaJ",
                                    "length": 376,
                                    "source_database": "reviewed",
                                    "sequence": "A" * 40 + "HPD" + "G" * 20,
                                },
                                "entries": [
                                    {
                                        "accession": "IPR001623",
                                        "entry_protein_locations": [
                                            {"fragments": [{"start": 41, "end": 43}]},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    jdp_rows, _fetches = classify_fetch_json(json.loads(dnaj_json.read_text(encoding="utf-8")), dnaj_rows="dedupe")
    write_classifications_csv(jdp_rows, jdp_csv)

    pocket_fields = [
        "accession",
        "confidence_tier",
        "mapping_confidence",
        "conservation_score",
        "mapping_mode",
        "contact_net_charge",
        "shell_net_charge",
        "delta_contact_net_charge",
        "delta_net_charge_vs_reference",
        "contact_mapping_fraction",
        "contact_sequence_identity",
        "mean_contact_ca_distance",
        "contact_drmsd",
        "n_contacts_within_5A",
        "sbd_alignment_coverage",
        "mean_plddt_pocket",
        "mapped_pocket_residues",
        "n_histidine",
        "net_charge_excluding_his",
        "exposed_net_charge",
        "charge_density",
        "quality_flags",
    ]
    with pocket_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pocket_fields)
        writer.writeheader()
        writer.writerow(dict.fromkeys(pocket_fields, ""))
        writer.writerow(
            {
                "accession": "P0A6Y8",
                "confidence_tier": "high",
                "mapping_confidence": "high",
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
                "quality_flags": "",
            },
        )

    with patch.object(
        sys,
        "argv",
        [
            "merge_all_features.py",
            "--dnak-json",
            str(dnak_json),
            "--dnaj-json",
            str(dnaj_json),
            "--pocket-csv",
            str(pocket_csv),
            "--jdp-csv",
            str(jdp_csv),
            "-o",
            str(output_csv),
        ],
    ):
        merge_all_main()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = {row["accession"]: row for row in csv.DictReader(handle)}

    assert set(rows) == {"P0A6Y8", "P0ACJ8"}
    assert rows["P0ACJ8"]["has_jdp_classification"] == "true"
    assert rows["P0ACJ8"]["jdp_predicted_class"] in {"A", "B", "C"}
    assert rows["P0A6Y8"]["has_pocket_charge"] == "true"
