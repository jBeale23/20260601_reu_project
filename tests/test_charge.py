"""Tests for structure_analysis pocket charge metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from structure_analysis.alignment import (
    PocketMappingInputs,
    expand_shell_residue_indices,
    map_reference_pocket,
)
from structure_analysis.analyze import (
    PocketChargeResult,
    _filter_results_for_summary,
    analyze_structure_file,
    merge_results_directory,
    result_from_dict,
    run_sensitivity_analysis,
    write_result,
)
from structure_analysis.charge import compute_pocket_metrics
from structure_analysis.geometry import apply_transform, kabsch_superpose
from structure_analysis.pdb_io import ResidueRecord, load_structure_residues, residues_to_sequence
from structure_analysis.pocket_reference import load_pocket_reference
from structure_analysis.quality import (
    MappingConfidenceInputs,
    QualityFlagInputs,
    assign_confidence_tier,
    assign_conservation_score,
    assign_mapping_confidence,
    build_quality_flags,
)
from structure_analysis.validation import derive_peptide_contact_resnums, load_pdb_chain


def _minimal_result_dict(
    *,
    accession: str = "P08113",
    mapping_confidence: str = "high",
    conservation_score: str = "low",
    confidence_tier: str = "low",
) -> dict:
    """Build a minimal serialized PocketChargeResult for filter/merge tests."""
    return {
        "accession": accession,
        "structure_path": f"data/dev_structures/AF-{accession}-F1-model_v6.pdb",
        "pocket_definition": "dnak_sbd_v1",
        "reference_accession": "P0A6Y8",
        "contact_residues_expected": 17,
        "contact_residues_mapped": 17,
        "contact_mapping_fraction": 1.0,
        "contact_sequence_identity": 0.18,
        "n_contact_mismatches": 10,
        "mapping_mode": "sbd_local",
        "sbd_alignment_coverage": 1.0,
        "contact_net_charge": 0,
        "shell_net_charge": 0,
        "n_positive": 1,
        "n_negative": 1,
        "n_histidine": 1,
        "net_charge": 0,
        "net_charge_excluding_his": -1,
        "exposed_net_charge": 0,
        "n_buried_charged": 0,
        "n_hydrophobic": 10,
        "mapped_pocket_residues": 50,
        "charge_density": 0.0,
        "mean_plddt_pocket": 85.0,
        "n_low_plddt_excluded": 0,
        "delta_contact_net_charge": 2,
        "delta_net_charge_vs_reference": 5,
        "mean_contact_ca_distance": 28.0,
        "max_contact_ca_distance": 35.0,
        "n_contacts_within_5A": 0,
        "sbd_superposition_rmsd": 25.0,
        "contact_drmsd": 8.7,
        "mapping_confidence": mapping_confidence,
        "conservation_score": conservation_score,
        "confidence_tier": confidence_tier,
        "quality_flags": ["charge_inversion_candidate"],
        "contact_residue_details": [],
        "warnings": [],
    }


def _minimal_result(**kwargs: object) -> PocketChargeResult:
    data = _minimal_result_dict()
    data.update(kwargs)
    return result_from_dict(data)


def _write_pdb(
    path: Path,
    residues: list[tuple[int, str, tuple[float, float, float], float | None]],
) -> None:
    """Write a minimal PDB with CA atoms. residues: (resnum, resname, coord, plddt)."""
    lines = []
    atom_id = 1
    for resnum, resname, (x, y, z), plddt in residues:
        bfactor = plddt if plddt is not None else 0.0
        lines.append(
            f"ATOM  {atom_id:5d}  CA  {resname:>3} A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C\n",
        )
        atom_id += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def mini_pdb(tmp_path: Path) -> Path:
    """Three-residue PDB with one charged residue."""
    path = tmp_path / "mini.pdb"
    _write_pdb(
        path,
        [
            (1, "ALA", (0.0, 0.0, 0.0), 90.0),
            (2, "LYS", (3.8, 0.0, 0.0), 85.0),
            (3, "GLU", (7.6, 0.0, 0.0), 80.0),
        ],
    )
    return path


def test_load_structure_residues_parses_plddt(mini_pdb: Path) -> None:
    """Parse CA atoms and AlphaFold pLDDT from B-factor column."""
    residues = load_structure_residues(mini_pdb)
    assert len(residues) == 3
    assert residues[1].resname == "LYS"
    assert residues[1].plddt == 85.0


def test_compute_pocket_metrics_counts_charges_and_filters_plddt() -> None:
    """Count charges, histidine, burial metrics, and exclude low pLDDT."""
    residues = [
        ResidueRecord(index=1, resnum=1, resname="LYS", ca_coord=(0.0, 0.0, 0.0), plddt=90.0),
        ResidueRecord(index=2, resnum=2, resname="ASP", ca_coord=(1.0, 0.0, 0.0), plddt=50.0),
        ResidueRecord(index=3, resnum=3, resname="LEU", ca_coord=(2.0, 0.0, 0.0), plddt=80.0),
        ResidueRecord(index=4, resnum=4, resname="ARG", ca_coord=(3.0, 0.0, 0.0), plddt=75.0),
        ResidueRecord(index=5, resnum=5, resname="HIS", ca_coord=(3.1, 0.0, 0.0), plddt=85.0),
    ]
    metrics = compute_pocket_metrics(residues, {1, 2, 3, 4, 5}, min_plddt=70.0)
    assert metrics.n_positive == 3
    assert metrics.n_negative == 0
    assert metrics.n_histidine == 1
    assert metrics.net_charge == 3
    assert metrics.net_charge_excluding_his == 2
    assert metrics.n_low_plddt_excluded == 1
    assert metrics.mapped_pocket_residues == 4


def test_expand_shell_residue_indices_respects_sbd_mask() -> None:
    """Shell expansion ignores residues outside the allowed SBD index set."""
    residues = [
        ResidueRecord(index=1, resnum=1, resname="ALA", ca_coord=(0.0, 0.0, 0.0), plddt=90.0),
        ResidueRecord(index=2, resnum=2, resname="ALA", ca_coord=(5.0, 0.0, 0.0), plddt=90.0),
        ResidueRecord(index=3, resnum=3, resname="ALA", ca_coord=(5.1, 0.0, 0.0), plddt=90.0),
    ]
    expanded = expand_shell_residue_indices(
        residues,
        {1},
        radius_angstrom=8.0,
        allowed_indices={1, 2},
    )
    assert expanded == {1, 2}
    assert 3 not in expanded


def test_map_reference_pocket(tmp_path: Path) -> None:
    """Alignment maps reference residue numbers onto target sequence indices."""
    ref_path = tmp_path / "ref.pdb"
    tgt_path = tmp_path / "tgt.pdb"
    _write_pdb(
        ref_path,
        [
            (10, "ALA", (0.0, 0.0, 0.0), 90.0),
            (11, "VAL", (3.8, 0.0, 0.0), 90.0),
            (12, "LEU", (7.6, 0.0, 0.0), 90.0),
        ],
    )
    _write_pdb(
        tgt_path,
        [
            (100, "ALA", (0.0, 0.0, 0.0), 90.0),
            (101, "VAL", (3.8, 0.0, 0.0), 90.0),
            (102, "LEU", (7.6, 0.0, 0.0), 90.0),
        ],
    )
    ref_residues = load_structure_residues(ref_path)
    tgt_residues = load_structure_residues(tgt_path)
    mapping = map_reference_pocket(
        PocketMappingInputs(
            reference_sequence=residues_to_sequence(ref_residues),
            target_sequence=residues_to_sequence(tgt_residues),
            reference_resnums=[11],
            reference_residues=ref_residues,
            target_residues=tgt_residues,
            sbd_residue_range=(10, 12),
        ),
    )
    assert mapping.warnings == []
    assert mapping.resnum_to_target_index[11] == 2
    assert mapping.n_contact_mismatches == 0
    assert mapping.mapping_mode in {"sbd_local", "full_length"}


def test_sbd_domain_alignment_maps_conserved_region(tmp_path: Path) -> None:
    """SBD-domain alignment maps contacts when N-terminal regions diverge."""
    ref_path = tmp_path / "ref.pdb"
    tgt_path = tmp_path / "tgt.pdb"
    ref_residues_data = [
        (1, "MET", (0.0, 0.0, 0.0), 90.0),
        (2, "GLY", (1.0, 0.0, 0.0), 90.0),
        (3, "PRO", (2.0, 0.0, 0.0), 90.0),
        (4, "SER", (3.0, 0.0, 0.0), 90.0),
        (5, "THR", (4.0, 0.0, 0.0), 90.0),
        (10, "ALA", (10.0, 0.0, 0.0), 90.0),
        (11, "VAL", (13.8, 0.0, 0.0), 90.0),
        (12, "LEU", (17.6, 0.0, 0.0), 90.0),
        (13, "ILE", (21.4, 0.0, 0.0), 90.0),
        (14, "PHE", (25.2, 0.0, 0.0), 90.0),
    ]
    tgt_residues_data = [
        (50, "CYS", (0.0, 0.0, 0.0), 90.0),
        (51, "ASP", (1.0, 0.0, 0.0), 90.0),
        (52, "GLU", (2.0, 0.0, 0.0), 90.0),
        (53, "ASN", (3.0, 0.0, 0.0), 90.0),
        (54, "GLN", (4.0, 0.0, 0.0), 90.0),
        (55, "ALA", (10.0, 0.0, 0.0), 90.0),
        (56, "VAL", (13.8, 0.0, 0.0), 90.0),
        (57, "LEU", (17.6, 0.0, 0.0), 90.0),
        (58, "ILE", (21.4, 0.0, 0.0), 90.0),
        (59, "PHE", (25.2, 0.0, 0.0), 90.0),
    ]
    _write_pdb(ref_path, ref_residues_data)
    _write_pdb(tgt_path, tgt_residues_data)
    ref_residues = load_structure_residues(ref_path)
    tgt_residues = load_structure_residues(tgt_path)
    ref_seq = residues_to_sequence(ref_residues)
    tgt_seq = residues_to_sequence(tgt_residues)

    mapping = map_reference_pocket(
        PocketMappingInputs(
            reference_sequence=ref_seq,
            target_sequence=tgt_seq,
            reference_resnums=[11],
            reference_residues=ref_residues,
            target_residues=tgt_residues,
            sbd_residue_range=(10, 14),
        ),
    )
    assert mapping.mapping_mode == "sbd_local"
    assert mapping.resnum_to_target_index[11] == 7
    assert mapping.n_contact_mismatches == 0


def test_kabsch_superpose_identity() -> None:
    """Kabsch superposition recovers a known rotation and translation."""
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    angle = np.pi / 4
    rotation_true = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    translation_true = np.array([5.0, -2.0, 3.0])
    tgt = (rotation_true @ ref.T).T + translation_true

    rotation, translation = kabsch_superpose(ref, tgt)
    aligned = apply_transform(tgt, rotation, translation)
    assert np.allclose(aligned, ref, atol=1e-5)


def test_assign_mapping_and_conservation_scores() -> None:
    """Mapping and conservation axes are assigned independently."""
    mapping = assign_mapping_confidence(
        MappingConfidenceInputs(
            contact_mapping_fraction=0.9,
            mean_plddt_pocket=80.0,
            mapped_pocket_residues=40,
            mean_contact_ca_distance=2.0,
            n_contacts_within_threshold=15,
            n_superposition_pairs=50,
            sbd_superposition_rmsd=3.0,
            contact_drmsd=1.5,
            mapping_mode="sbd_local",
            sbd_alignment_coverage=0.95,
        ),
    )
    assert mapping == "high"
    assert assign_conservation_score(0.8, 0) == "high"
    assert assign_conservation_score(0.5, 5) == "medium"
    assert assign_conservation_score(0.2, 10) == "low"
    assert assign_confidence_tier("high", "low") == "low"
    assert assign_confidence_tier("high", "high") == "high"


def test_assign_confidence_tier() -> None:
    """Combined tier is the conservative minimum of mapping and conservation."""
    assert assign_confidence_tier("high", "high") == "high"
    assert assign_confidence_tier("high", "medium") == "medium"
    assert assign_confidence_tier("high", "low") == "low"
    assert assign_confidence_tier("medium", "low") == "low"


def test_build_quality_flags() -> None:
    """Quality flags highlight mapping, geometry, identity, and inversion signals."""
    flags = build_quality_flags(
        QualityFlagInputs(
            contact_mapping_fraction=0.5,
            mean_plddt_pocket=60.0,
            mapped_pocket_residues=20,
            n_contact_mismatches=5,
            n_low_plddt_excluded=3,
            mapping_mode="full_length",
            sbd_alignment_coverage=0.3,
            contact_sequence_identity=0.35,
            mean_contact_ca_distance=8.0,
            contact_drmsd=7.0,
            sbd_superposition_rmsd=20.0,
            mapping_confidence="high",
            conservation_score="low",
            delta_contact_net_charge=2,
        ),
    )
    assert "full_length_alignment" in flags
    assert "low_sbd_coverage" in flags
    assert "unreliable_superposition" in flags
    assert "low_contact_mapping" in flags
    assert "few_pocket_residues" in flags
    assert "low_plddt" in flags
    assert "poor_structural_superposition" in flags
    assert "poor_contact_geometry" in flags
    assert "divergent_contact_sequences" in flags
    assert "low_contact_identity" in flags
    assert "contact_mismatches" in flags
    assert "many_contact_mismatches" in flags
    assert "plddt_filtered" in flags
    assert "charge_inversion_candidate" in flags


def test_load_pocket_reference_v2_fields() -> None:
    """Load pocket YAML including SBD range, min_plddt, and max_contact_ca_distance."""
    path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    pocket = load_pocket_reference(path)
    assert pocket.reference_accession == "P0A6Y8"
    assert pocket.sbd_residue_range == (386, 638)
    assert pocket.min_plddt == 70.0
    assert pocket.max_contact_ca_distance == 5.0
    assert len(pocket.contact_residues) == 17


@pytest.mark.skipif(
    not Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb").is_file(),
    reason="Dev AlphaFold structures not downloaded",
)
def test_analyze_ecoli_dnak_reference() -> None:
    """Reference DnaK maps pocket residues and yields zero delta vs itself."""
    pocket_path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    ref_pdb = Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb")
    result = analyze_structure_file(ref_pdb, pocket_path, ref_pdb)
    assert result.accession == "P0A6Y8"
    assert result.mapped_pocket_residues > 0
    assert result.delta_net_charge_vs_reference == 0
    assert result.delta_contact_net_charge == 0
    assert result.confidence_tier == "high"
    assert result.mapping_confidence == "high"
    assert result.conservation_score == "high"
    assert result.mapping_mode == "sbd_local"
    assert result.mean_contact_ca_distance == 0.0
    assert len(result.contact_residue_details) == 17


@pytest.mark.skipif(
    not Path("data/dev_structures/AF-P08113-F1-model_v6.pdb").is_file(),
    reason="Dev AlphaFold structures not downloaded",
)
def test_p08113_mapping_high_conservation_low() -> None:
    """Yeast HSP70: reliable mapping with divergent contact sequences and charge shift."""
    pocket_path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    ref_pdb = Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb")
    tgt_pdb = Path("data/dev_structures/AF-P08113-F1-model_v6.pdb")
    result = analyze_structure_file(tgt_pdb, pocket_path, ref_pdb)
    assert result.mapping_confidence == "high"
    assert result.conservation_score == "low"
    assert result.delta_contact_net_charge == 2
    assert "charge_inversion_candidate" in result.quality_flags


@pytest.mark.skipif(
    not Path("data/dev_structures/AF-P61889-F1-model_v6.pdb").is_file(),
    reason="Dev AlphaFold structures not downloaded",
)
def test_p61889_poor_structural_mapping() -> None:
    """Divergent thermophile DnaK should have low mapping confidence."""
    pocket_path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    ref_pdb = Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb")
    tgt_pdb = Path("data/dev_structures/AF-P61889-F1-model_v6.pdb")
    result = analyze_structure_file(tgt_pdb, pocket_path, ref_pdb)
    assert result.mapping_confidence == "low"
    assert result.conservation_score == "low"


@pytest.mark.skipif(
    not Path("tests/fixtures/1DKX.pdb").is_file(),
    reason="1DKX fixture not downloaded",
)
def test_1dkx_derived_contacts_match_yaml() -> None:
    """Independently derived 1DKX contacts match the pocket YAML definition."""
    pocket = load_pocket_reference(Path("data/pocket_refs/dnak_sbd_pocket.yaml"))
    fixture = Path("tests/fixtures/1DKX.pdb")
    protein = load_pdb_chain(fixture, "A")
    peptide = load_pdb_chain(fixture, "B")
    derived = derive_peptide_contact_resnums(
        protein,
        peptide,
        radius_angstrom=pocket.shell_radius_angstrom,
    )
    assert set(derived) == set(pocket.contact_residues)


@pytest.mark.skipif(
    not Path("tests/fixtures/1DKX.pdb").is_file(),
    not Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb").is_file(),
    reason="1DKX fixture or dev structures not available",
)
def test_alphafold_vs_1dkx_experimental_comparison() -> None:
    """AlphaFold and experimental reference structures both analyze successfully."""
    pocket_path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    af_pdb = Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb")
    af_result = analyze_structure_file(af_pdb, pocket_path, af_pdb)
    assert af_result.confidence_tier == "high"
    assert af_result.contact_residues_mapped == 17


@pytest.mark.skipif(
    not Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb").is_file(),
    reason="Dev AlphaFold structures not downloaded",
)
def test_sensitivity_analysis_writes_report(tmp_path: Path) -> None:
    """Sensitivity sweep produces report and stability summary CSV."""
    structures = Path("data/dev_structures")
    pocket_path = Path("data/pocket_refs/dnak_sbd_pocket.yaml")
    ref_pdb = Path("data/dev_structures/AF-P0A6Y8-F1-model_v6.pdb")
    out_dir = tmp_path / "sensitivity"
    report = run_sensitivity_analysis(structures, pocket_path, ref_pdb, out_dir)
    assert report.is_file()
    assert (out_dir / "sensitivity_stability.csv").is_file()
    content = report.read_text(encoding="utf-8")
    assert "shell_radius" in content
    assert "P0A6Y8" in content


def test_filter_results_min_mapping_vs_combined_confidence() -> None:
    """Mapping-only filter keeps charge-inversion candidates excluded by combined tier."""
    inversion = _minimal_result(
        mapping_confidence="high",
        conservation_score="low",
        confidence_tier="low",
    )
    neutral = _minimal_result(
        accession="P11142",
        mapping_confidence="high",
        conservation_score="high",
        confidence_tier="high",
    )
    results = [inversion, neutral]

    mapping_high = _filter_results_for_summary(results, min_mapping_confidence="high")
    assert {r.accession for r in mapping_high} == {"P08113", "P11142"}

    combined_high = _filter_results_for_summary(results, min_confidence="high")
    assert {r.accession for r in combined_high} == {"P11142"}


def test_merge_results_directory_roundtrip(tmp_path: Path) -> None:
    """merge_results_directory loads JSON files and writes summary CSV."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = _minimal_result()
    write_result(result, results_dir / "pocket_charge_P08113.json")

    merged = merge_results_directory(results_dir, results_dir, min_mapping_confidence="high")

    assert len(merged) == 1
    csv_path = results_dir / "pocket_charge_summary.csv"
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "P08113" in content
    assert "mapping_confidence" in content.splitlines()[0]


def test_merge_results_directory_ignores_summary_json(tmp_path: Path) -> None:
    """Re-merge must skip pocket_charge_summary.json left from a prior merge."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = _minimal_result()
    write_result(result, results_dir / "pocket_charge_P08113.json")

    merge_results_directory(results_dir, results_dir)
    assert (results_dir / "pocket_charge_summary.json").is_file()

    merged_again = merge_results_directory(results_dir, results_dir)
    assert len(merged_again) == 1
    assert merged_again[0].accession == "P08113"
