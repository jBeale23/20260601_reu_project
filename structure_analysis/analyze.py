"""Analyze binding-pocket charge on AlphaFold structures."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from structure_analysis.alignment import (
    MappingMode,
    PocketMappingInputs,
    expand_shell_residue_indices,
    map_reference_pocket,
)
from structure_analysis.charge import (
    PocketChargeMetrics,
    charge_density,
    compute_pocket_metrics,
    residue_charge_contribution,
)
from structure_analysis.geometry import StructuralMetricsInputs, compute_structural_metrics
from structure_analysis.pdb_io import (
    load_structure_residues,
    parse_accession_from_path,
    residue_by_index,
    residues_to_sequence,
)
from structure_analysis.pocket_reference import PocketReference, load_pocket_reference
from structure_analysis.quality import (
    ConfidenceTier,
    MappingConfidenceInputs,
    QualityFlagInputs,
    assign_confidence_tier,
    assign_conservation_score,
    assign_mapping_confidence,
    build_quality_flags,
    tier_rank,
)


@dataclass(frozen=True, slots=True)
class ContactResidueDetail:
    """Per-contact residue charge attribution."""

    reference_resnum: int
    target_index: int
    target_resnum: int
    target_resname: str
    reference_resname: str
    charge_contribution: int
    mismatch: bool
    ca_distance_angstrom: float | None


@dataclass(frozen=True, slots=True)
class PocketChargeResult:
    """Full result for one structure."""

    accession: str
    structure_path: str
    pocket_definition: str
    reference_accession: str
    contact_residues_expected: int
    contact_residues_mapped: int
    contact_mapping_fraction: float
    contact_sequence_identity: float
    n_contact_mismatches: int
    mapping_mode: MappingMode
    sbd_alignment_coverage: float
    contact_net_charge: int
    shell_net_charge: int
    n_positive: int
    n_negative: int
    n_histidine: int
    net_charge: int
    net_charge_excluding_his: int
    exposed_net_charge: int
    n_buried_charged: int
    n_hydrophobic: int
    mapped_pocket_residues: int
    charge_density: float | None
    mean_plddt_pocket: float | None
    n_low_plddt_excluded: int
    delta_contact_net_charge: int | None
    delta_net_charge_vs_reference: int | None
    mean_contact_ca_distance: float | None
    max_contact_ca_distance: float | None
    n_contacts_within_5A: int  # noqa: N815 — CSV column name uses angstrom suffix
    sbd_superposition_rmsd: float | None
    contact_drmsd: float | None
    mapping_confidence: ConfidenceTier
    conservation_score: ConfidenceTier
    confidence_tier: ConfidenceTier
    quality_flags: list[str]
    contact_residue_details: list[ContactResidueDetail]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class ReferenceContext:
    """Cached reference structure data for batch analysis."""

    pocket_ref: PocketReference
    reference_residues: list
    reference_sequence: str
    contact_net_charge: int
    shell_net_charge: int


@dataclass(frozen=True, slots=True)
class ContactDetailInputs:
    """Inputs for per-contact residue charge attribution."""

    mapping: dict[int, int]
    reference_resnums: list[int]
    reference_residues: list
    target_residues: list
    reference_sequence: str
    target_sequence: str
    contact_ca_distances: dict[int, float]


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Inputs for pocket charge analysis on one structure."""

    structure_path: Path
    accession: str
    target_residues: list
    target_sequence: str
    pocket_ref: PocketReference
    reference_residues: list
    reference_sequence: str
    reference_contact_charge: int | None
    reference_shell_charge: int | None


def _build_contact_residue_details(inputs: ContactDetailInputs) -> list[ContactResidueDetail]:
    ref_by_index = residue_by_index(inputs.reference_residues)
    ref_resnum_to_seq_index = {r.resnum: r.index for r in inputs.reference_residues}
    tgt_by_index = residue_by_index(inputs.target_residues)
    details: list[ContactResidueDetail] = []

    for resnum in inputs.reference_resnums:
        tgt_index = inputs.mapping.get(resnum)
        if tgt_index is None:
            continue
        ref_seq_index = ref_resnum_to_seq_index.get(resnum)
        ref_residue = ref_by_index.get(ref_seq_index) if ref_seq_index else None
        tgt_residue = tgt_by_index.get(tgt_index)
        if tgt_residue is None:
            continue
        ref_resname = ref_residue.resname if ref_residue else "UNK"
        ref_char = inputs.reference_sequence[ref_seq_index - 1] if ref_seq_index else "X"
        tgt_char = inputs.target_sequence[tgt_index - 1]
        details.append(
            ContactResidueDetail(
                reference_resnum=resnum,
                target_index=tgt_index,
                target_resnum=tgt_residue.resnum,
                target_resname=tgt_residue.resname,
                reference_resname=ref_resname,
                charge_contribution=residue_charge_contribution(tgt_residue.resname),
                mismatch=ref_char != tgt_char,
                ca_distance_angstrom=inputs.contact_ca_distances.get(resnum),
            ),
        )
    return details


def _build_reference_context(
    pocket_ref: PocketReference,
    reference_structure_path: Path,
) -> ReferenceContext:
    reference_residues = load_structure_residues(reference_structure_path)
    reference_sequence = residues_to_sequence(reference_residues)
    ref_result = _analyze_with_residues(
        AnalysisInputs(
            structure_path=reference_structure_path,
            accession=pocket_ref.reference_accession,
            target_residues=reference_residues,
            target_sequence=reference_sequence,
            pocket_ref=pocket_ref,
            reference_residues=reference_residues,
            reference_sequence=reference_sequence,
            reference_contact_charge=None,
            reference_shell_charge=None,
        ),
    )
    return ReferenceContext(
        pocket_ref=pocket_ref,
        reference_residues=reference_residues,
        reference_sequence=reference_sequence,
        contact_net_charge=ref_result.contact_net_charge,
        shell_net_charge=ref_result.shell_net_charge,
    )


def _analyze_with_residues(inputs: AnalysisInputs) -> PocketChargeResult:
    contact_resnums = list(inputs.pocket_ref.contact_residues)
    mapping = map_reference_pocket(
        PocketMappingInputs(
            reference_sequence=inputs.reference_sequence,
            target_sequence=inputs.target_sequence,
            reference_resnums=contact_resnums,
            reference_residues=inputs.reference_residues,
            target_residues=inputs.target_residues,
            sbd_residue_range=inputs.pocket_ref.sbd_residue_range,
        ),
    )
    warnings = list(mapping.warnings)

    contact_indices = set(mapping.resnum_to_target_index.values())
    shell_indices = expand_shell_residue_indices(
        inputs.target_residues,
        contact_indices,
        inputs.pocket_ref.shell_radius_angstrom,
        allowed_indices=mapping.sbd_target_indices,
    )

    contact_metrics: PocketChargeMetrics = compute_pocket_metrics(
        inputs.target_residues,
        contact_indices,
        min_plddt=inputs.pocket_ref.min_plddt,
    )
    shell_metrics: PocketChargeMetrics = compute_pocket_metrics(
        inputs.target_residues,
        shell_indices,
        min_plddt=inputs.pocket_ref.min_plddt,
    )

    contact_expected = len(contact_resnums)
    contact_mapped = len(mapping.resnum_to_target_index)
    contact_fraction = contact_mapped / contact_expected if contact_expected else 0.0

    delta_contact = None
    delta_shell = None
    if inputs.reference_contact_charge is not None:
        delta_contact = contact_metrics.net_charge - inputs.reference_contact_charge
    if inputs.reference_shell_charge is not None:
        delta_shell = shell_metrics.net_charge - inputs.reference_shell_charge

    structural = compute_structural_metrics(
        StructuralMetricsInputs(
            reference_residues=inputs.reference_residues,
            target_residues=inputs.target_residues,
            ref_index_to_target_index=mapping.ref_index_to_target_index,
            resnum_to_target_index=mapping.resnum_to_target_index,
            sbd_target_indices=mapping.sbd_target_indices,
            contact_resnums=contact_resnums,
            max_contact_ca_distance=inputs.pocket_ref.max_contact_ca_distance,
        ),
    )

    mapping_confidence = assign_mapping_confidence(
        MappingConfidenceInputs(
            contact_mapping_fraction=contact_fraction,
            mean_plddt_pocket=shell_metrics.mean_plddt_pocket,
            mapped_pocket_residues=shell_metrics.mapped_pocket_residues,
            mean_contact_ca_distance=structural.mean_contact_ca_distance,
            n_contacts_within_threshold=structural.n_contacts_within_threshold,
            n_superposition_pairs=structural.n_superposition_pairs,
            sbd_superposition_rmsd=structural.sbd_superposition_rmsd,
            contact_drmsd=structural.contact_drmsd,
            mapping_mode=mapping.mapping_mode,
            sbd_alignment_coverage=mapping.sbd_alignment_coverage,
        ),
    )
    conservation_score = assign_conservation_score(
        mapping.contact_sequence_identity,
        mapping.n_contact_mismatches,
    )
    confidence_tier = assign_confidence_tier(mapping_confidence, conservation_score)
    quality_flags = build_quality_flags(
        QualityFlagInputs(
            contact_mapping_fraction=contact_fraction,
            mean_plddt_pocket=shell_metrics.mean_plddt_pocket,
            mapped_pocket_residues=shell_metrics.mapped_pocket_residues,
            n_contact_mismatches=mapping.n_contact_mismatches,
            n_low_plddt_excluded=shell_metrics.n_low_plddt_excluded,
            mapping_mode=mapping.mapping_mode,
            sbd_alignment_coverage=mapping.sbd_alignment_coverage,
            contact_sequence_identity=mapping.contact_sequence_identity,
            mean_contact_ca_distance=structural.mean_contact_ca_distance,
            contact_drmsd=structural.contact_drmsd,
            sbd_superposition_rmsd=structural.sbd_superposition_rmsd,
            mapping_confidence=mapping_confidence,
            conservation_score=conservation_score,
            delta_contact_net_charge=delta_contact,
        ),
    )

    contact_details = _build_contact_residue_details(
        ContactDetailInputs(
            mapping=mapping.resnum_to_target_index,
            reference_resnums=contact_resnums,
            reference_residues=inputs.reference_residues,
            target_residues=inputs.target_residues,
            reference_sequence=inputs.reference_sequence,
            target_sequence=inputs.target_sequence,
            contact_ca_distances=structural.contact_ca_distances,
        ),
    )

    return PocketChargeResult(
        accession=inputs.accession,
        structure_path=str(inputs.structure_path),
        pocket_definition=inputs.pocket_ref.pocket_definition,
        reference_accession=inputs.pocket_ref.reference_accession,
        contact_residues_expected=contact_expected,
        contact_residues_mapped=contact_mapped,
        contact_mapping_fraction=round(contact_fraction, 4),
        contact_sequence_identity=round(mapping.contact_sequence_identity, 4),
        n_contact_mismatches=mapping.n_contact_mismatches,
        mapping_mode=mapping.mapping_mode,
        sbd_alignment_coverage=mapping.sbd_alignment_coverage,
        contact_net_charge=contact_metrics.net_charge,
        shell_net_charge=shell_metrics.net_charge,
        n_positive=shell_metrics.n_positive,
        n_negative=shell_metrics.n_negative,
        n_histidine=shell_metrics.n_histidine,
        net_charge=shell_metrics.net_charge,
        net_charge_excluding_his=shell_metrics.net_charge_excluding_his,
        exposed_net_charge=shell_metrics.exposed_net_charge,
        n_buried_charged=shell_metrics.n_buried_charged,
        n_hydrophobic=shell_metrics.n_hydrophobic,
        mapped_pocket_residues=shell_metrics.mapped_pocket_residues,
        charge_density=charge_density(shell_metrics.net_charge, shell_metrics.mapped_pocket_residues),
        mean_plddt_pocket=(
            round(shell_metrics.mean_plddt_pocket, 2) if shell_metrics.mean_plddt_pocket is not None else None
        ),
        n_low_plddt_excluded=shell_metrics.n_low_plddt_excluded,
        delta_contact_net_charge=delta_contact,
        delta_net_charge_vs_reference=delta_shell,
        mean_contact_ca_distance=(
            round(structural.mean_contact_ca_distance, 2) if structural.mean_contact_ca_distance is not None else None
        ),
        max_contact_ca_distance=(
            round(structural.max_contact_ca_distance, 2) if structural.max_contact_ca_distance is not None else None
        ),
        n_contacts_within_5A=structural.n_contacts_within_threshold,
        sbd_superposition_rmsd=(
            round(structural.sbd_superposition_rmsd, 2) if structural.sbd_superposition_rmsd is not None else None
        ),
        contact_drmsd=(round(structural.contact_drmsd, 2) if structural.contact_drmsd is not None else None),
        mapping_confidence=mapping_confidence,
        conservation_score=conservation_score,
        confidence_tier=confidence_tier,
        quality_flags=quality_flags,
        contact_residue_details=contact_details,
        warnings=warnings,
    )


def analyze_structure(
    structure_path: Path,
    context: ReferenceContext,
) -> PocketChargeResult:
    """Compute pocket charge metrics for one PDB using cached reference context."""
    target_residues = load_structure_residues(structure_path)
    target_sequence = residues_to_sequence(target_residues)
    accession = parse_accession_from_path(structure_path) or structure_path.stem
    return _analyze_with_residues(
        AnalysisInputs(
            structure_path=structure_path,
            accession=accession,
            target_residues=target_residues,
            target_sequence=target_sequence,
            pocket_ref=context.pocket_ref,
            reference_residues=context.reference_residues,
            reference_sequence=context.reference_sequence,
            reference_contact_charge=context.contact_net_charge,
            reference_shell_charge=context.shell_net_charge,
        ),
    )


def analyze_structure_file(
    structure_path: Path,
    pocket_ref_path: Path,
    reference_structure_path: Path,
    pocket_ref: PocketReference | None = None,
) -> PocketChargeResult:
    """Convenience wrapper for single-structure analysis."""
    pocket_ref = pocket_ref or load_pocket_reference(pocket_ref_path)
    context = _build_reference_context(pocket_ref, reference_structure_path)
    return analyze_structure(structure_path, context)


def _result_to_dict(result: PocketChargeResult) -> dict[str, Any]:
    data = asdict(result)
    data["contact_residue_details"] = [asdict(d) for d in result.contact_residue_details]
    return data


def result_from_dict(data: dict[str, Any]) -> PocketChargeResult:
    """Deserialize a PocketChargeResult written by write_result."""
    details = [ContactResidueDetail(**detail) for detail in data["contact_residue_details"]]
    return PocketChargeResult(
        accession=str(data["accession"]),
        structure_path=str(data["structure_path"]),
        pocket_definition=str(data["pocket_definition"]),
        reference_accession=str(data["reference_accession"]),
        contact_residues_expected=int(data["contact_residues_expected"]),
        contact_residues_mapped=int(data["contact_residues_mapped"]),
        contact_mapping_fraction=float(data["contact_mapping_fraction"]),
        contact_sequence_identity=float(data["contact_sequence_identity"]),
        n_contact_mismatches=int(data["n_contact_mismatches"]),
        mapping_mode=data["mapping_mode"],
        sbd_alignment_coverage=float(data["sbd_alignment_coverage"]),
        contact_net_charge=int(data["contact_net_charge"]),
        shell_net_charge=int(data["shell_net_charge"]),
        n_positive=int(data["n_positive"]),
        n_negative=int(data["n_negative"]),
        n_histidine=int(data["n_histidine"]),
        net_charge=int(data["net_charge"]),
        net_charge_excluding_his=int(data["net_charge_excluding_his"]),
        exposed_net_charge=int(data["exposed_net_charge"]),
        n_buried_charged=int(data["n_buried_charged"]),
        n_hydrophobic=int(data["n_hydrophobic"]),
        mapped_pocket_residues=int(data["mapped_pocket_residues"]),
        charge_density=data["charge_density"],
        mean_plddt_pocket=data["mean_plddt_pocket"],
        n_low_plddt_excluded=int(data["n_low_plddt_excluded"]),
        delta_contact_net_charge=data["delta_contact_net_charge"],
        delta_net_charge_vs_reference=data["delta_net_charge_vs_reference"],
        mean_contact_ca_distance=data["mean_contact_ca_distance"],
        max_contact_ca_distance=data["max_contact_ca_distance"],
        n_contacts_within_5A=int(data["n_contacts_within_5A"]),
        sbd_superposition_rmsd=data["sbd_superposition_rmsd"],
        contact_drmsd=data["contact_drmsd"],
        mapping_confidence=data["mapping_confidence"],
        conservation_score=data["conservation_score"],
        confidence_tier=data["confidence_tier"],
        quality_flags=list(data["quality_flags"]),
        contact_residue_details=details,
        warnings=list(data["warnings"]),
    )


def load_result_from_json(path: Path) -> PocketChargeResult:
    """Load one per-accession pocket charge JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return result_from_dict(data)


def _filter_results_for_summary(
    results: list[PocketChargeResult],
    *,
    min_confidence: ConfidenceTier | None = None,
    min_mapping_confidence: ConfidenceTier | None = None,
) -> list[PocketChargeResult]:
    """Apply optional combined-tier and mapping-tier filters for summary output."""
    filtered = results
    if min_confidence is not None:
        min_rank = tier_rank(min_confidence)
        filtered = [r for r in filtered if tier_rank(r.confidence_tier) >= min_rank]
    if min_mapping_confidence is not None:
        min_rank = tier_rank(min_mapping_confidence)
        filtered = [r for r in filtered if tier_rank(r.mapping_confidence) >= min_rank]
    return filtered


def _write_summary_files(
    results: list[PocketChargeResult],
    output_dir: Path,
) -> None:
    """Write pocket_charge_summary.json and pocket_charge_summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "pocket_charge_summary.json"
    summary_json.write_text(
        json.dumps([_result_to_dict(r) for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv_summary(results, output_dir / "pocket_charge_summary.csv")


def write_result(result: PocketChargeResult, output_path: Path) -> None:
    """Write analysis result as JSON."""
    output_path.write_text(json.dumps(_result_to_dict(result), indent=2) + "\n", encoding="utf-8")


CSV_COLUMNS = [
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


def write_csv_summary(results: list[PocketChargeResult], output_path: Path) -> None:
    """Write a mentor-friendly CSV summary table."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "accession": result.accession,
                    "confidence_tier": result.confidence_tier,
                    "mapping_confidence": result.mapping_confidence,
                    "conservation_score": result.conservation_score,
                    "mapping_mode": result.mapping_mode,
                    "contact_net_charge": result.contact_net_charge,
                    "shell_net_charge": result.shell_net_charge,
                    "delta_contact_net_charge": result.delta_contact_net_charge,
                    "delta_net_charge_vs_reference": result.delta_net_charge_vs_reference,
                    "contact_mapping_fraction": result.contact_mapping_fraction,
                    "contact_sequence_identity": result.contact_sequence_identity,
                    "mean_contact_ca_distance": result.mean_contact_ca_distance,
                    "contact_drmsd": result.contact_drmsd,
                    "n_contacts_within_5A": result.n_contacts_within_5A,
                    "sbd_alignment_coverage": result.sbd_alignment_coverage,
                    "mean_plddt_pocket": result.mean_plddt_pocket,
                    "mapped_pocket_residues": result.mapped_pocket_residues,
                    "n_histidine": result.n_histidine,
                    "net_charge_excluding_his": result.net_charge_excluding_his,
                    "exposed_net_charge": result.exposed_net_charge,
                    "charge_density": result.charge_density,
                    "quality_flags": ";".join(result.quality_flags),
                },
            )


def export_pocket_residues(
    structure_path: Path,
    pocket_ref_path: Path,
    reference_structure_path: Path,
    output_path: Path,
) -> None:
    """Export per-residue pocket membership for PyMOL validation."""
    pocket_ref = load_pocket_reference(pocket_ref_path)
    context = _build_reference_context(pocket_ref, reference_structure_path)
    target_residues = load_structure_residues(structure_path)
    target_sequence = residues_to_sequence(target_residues)
    mapping = map_reference_pocket(
        PocketMappingInputs(
            reference_sequence=context.reference_sequence,
            target_sequence=target_sequence,
            reference_resnums=list(pocket_ref.contact_residues),
            reference_residues=context.reference_residues,
            target_residues=target_residues,
            sbd_residue_range=pocket_ref.sbd_residue_range,
        ),
    )
    contact_indices = set(mapping.resnum_to_target_index.values())
    shell_indices = expand_shell_residue_indices(
        target_residues,
        contact_indices,
        pocket_ref.shell_radius_angstrom,
        allowed_indices=mapping.sbd_target_indices,
    )

    lines = ["index,resnum,resname,plddt,in_contact,in_shell\n"]
    lines.extend(
        (
            f"{residue.index},{residue.resnum},{residue.resname},"
            f"{residue.plddt if residue.plddt is not None else ''},"
            f"{int(residue.index in contact_indices)},"
            f"{int(residue.index in shell_indices)}\n"
        )
        for residue in target_residues
    )
    output_path.write_text("".join(lines), encoding="utf-8")


def export_contact_attribution_csv(result: PocketChargeResult, output_path: Path) -> None:
    """Write per-contact residue charge attribution CSV."""
    fieldnames = [
        "reference_resnum",
        "target_resnum",
        "target_resname",
        "reference_resname",
        "charge_contribution",
        "mismatch",
        "ca_distance_angstrom",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detail in result.contact_residue_details:
            writer.writerow(asdict(detail))


@dataclass(frozen=True, slots=True)
class BatchAnalysisOptions:
    """Options for directory batch pocket charge analysis."""

    structures_dir: Path
    pocket_ref_path: Path
    reference_structure_path: Path
    output_dir: Path
    min_confidence: ConfidenceTier | None = None
    min_mapping_confidence: ConfidenceTier | None = None


def analyze_directory(options: BatchAnalysisOptions) -> list[PocketChargeResult]:
    """Analyze all PDB files in a directory using cached reference context."""
    pocket_ref = load_pocket_reference(options.pocket_ref_path)
    context = _build_reference_context(pocket_ref, options.reference_structure_path)

    options.output_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = sorted(options.structures_dir.glob("*.pdb")) + sorted(options.structures_dir.glob("*.pdb.gz"))
    all_results: list[PocketChargeResult] = []
    for pdb_path in pdb_paths:
        result = analyze_structure(pdb_path, context)
        all_results.append(result)
        write_result(result, options.output_dir / f"pocket_charge_{result.accession}.json")

    summary_results = _filter_results_for_summary(
        all_results,
        min_confidence=options.min_confidence,
        min_mapping_confidence=options.min_mapping_confidence,
    )
    _write_summary_files(summary_results, options.output_dir)
    return all_results


def _per_accession_result_json_paths(results_dir: Path) -> list[Path]:
    """Return per-accession result JSON paths, excluding aggregate summary files."""
    return sorted(
        path for path in results_dir.glob("pocket_charge_*.json") if path.name != "pocket_charge_summary.json"
    )


def merge_results_directory(
    results_dir: Path,
    output_dir: Path | None = None,
    *,
    min_confidence: ConfidenceTier | None = None,
    min_mapping_confidence: ConfidenceTier | None = None,
) -> list[PocketChargeResult]:
    """Load pocket_charge_*.json files and write summary CSV/JSON."""
    json_paths = _per_accession_result_json_paths(results_dir)
    if not json_paths:
        msg = f"No pocket_charge_*.json files found in {results_dir}"
        raise ValueError(msg)

    all_results = [load_result_from_json(path) for path in json_paths]
    summary_results = _filter_results_for_summary(
        all_results,
        min_confidence=min_confidence,
        min_mapping_confidence=min_mapping_confidence,
    )
    _write_summary_files(summary_results, output_dir or results_dir)
    return all_results


SENSITIVITY_SHELL_RADII = (6.0, 8.0, 10.0)
SENSITIVITY_MIN_PLDDT = (60.0, 70.0, 80.0)
SENSITIVITY_SBD_PADDING = (0, 10)


def run_sensitivity_analysis(
    structures_dir: Path,
    pocket_ref_path: Path,
    reference_structure_path: Path,
    output_dir: Path,
) -> Path:
    """Sweep shell radius, min_plddt, and SBD window; write stability report CSV."""
    base_ref = load_pocket_reference(pocket_ref_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sensitivity_report.csv"

    rows: list[dict[str, Any]] = []
    for shell_radius in SENSITIVITY_SHELL_RADII:
        for min_plddt in SENSITIVITY_MIN_PLDDT:
            for sbd_pad in SENSITIVITY_SBD_PADDING:
                sbd_start = base_ref.sbd_residue_range[0] - sbd_pad
                sbd_end = base_ref.sbd_residue_range[1] + sbd_pad
                pocket_ref = PocketReference(
                    pocket_definition=base_ref.pocket_definition,
                    reference_accession=base_ref.reference_accession,
                    reference_pdb=base_ref.reference_pdb,
                    reference_pdb_url=base_ref.reference_pdb_url,
                    shell_radius_angstrom=shell_radius,
                    sbd_residue_range=(sbd_start, sbd_end),
                    min_plddt=min_plddt,
                    max_contact_ca_distance=base_ref.max_contact_ca_distance,
                    contact_residues=base_ref.contact_residues,
                )
                context = _build_reference_context(pocket_ref, reference_structure_path)
                for pdb_path in sorted(structures_dir.glob("*.pdb")) + sorted(structures_dir.glob("*.pdb.gz")):
                    result = analyze_structure(pdb_path, context)
                    rows.append(
                        {
                            "accession": result.accession,
                            "shell_radius": shell_radius,
                            "min_plddt": min_plddt,
                            "sbd_padding": sbd_pad,
                            "delta_contact": result.delta_contact_net_charge,
                            "delta_shell": result.delta_net_charge_vs_reference,
                            "confidence_tier": result.confidence_tier,
                            "contact_sequence_identity": result.contact_sequence_identity,
                        },
                    )

    fieldnames = [
        "accession",
        "shell_radius",
        "min_plddt",
        "sbd_padding",
        "delta_contact",
        "delta_shell",
        "confidence_tier",
        "contact_sequence_identity",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stability_path = output_dir / "sensitivity_stability.csv"
    _write_stability_summary(rows, stability_path)
    return report_path


def _write_stability_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Summarize whether charge deltas are stable across parameter sweeps."""
    by_accession: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_accession.setdefault(str(row["accession"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for accession, acc_rows in sorted(by_accession.items()):
        contact_deltas = [r["delta_contact"] for r in acc_rows if r["delta_contact"] is not None]
        shell_deltas = [r["delta_shell"] for r in acc_rows if r["delta_shell"] is not None]
        summary_rows.append(
            {
                "accession": accession,
                "contact_delta_min": min(contact_deltas) if contact_deltas else "",
                "contact_delta_max": max(contact_deltas) if contact_deltas else "",
                "shell_delta_min": min(shell_deltas) if shell_deltas else "",
                "shell_delta_max": max(shell_deltas) if shell_deltas else "",
                "stable_contact": (min(contact_deltas) == max(contact_deltas) if contact_deltas else False),
                "stable_shell": min(shell_deltas) == max(shell_deltas) if shell_deltas else False,
                "n_sweeps": len(acc_rows),
            },
        )

    fieldnames = [
        "accession",
        "contact_delta_min",
        "contact_delta_max",
        "shell_delta_min",
        "shell_delta_max",
        "stable_contact",
        "stable_shell",
        "n_sweeps",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
