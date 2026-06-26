"""Confidence tiers and quality flags for pocket charge results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from structure_analysis.alignment import MIN_SBD_ALIGNMENT_COVERAGE, MappingMode
from structure_analysis.geometry import MIN_SBD_PAIRS_FOR_SUPERPOSITION, RELIABLE_SUPERPOSITION_RMSD

ConfidenceTier = Literal["high", "medium", "low"]

HIGH_CONTACT_MAPPING = 0.85
MEDIUM_CONTACT_MAPPING = 0.60
MIN_MAPPED_POCKET_RESIDUES = 30
MIN_MEAN_PLDDT = 70.0
HIGH_CONTACT_IDENTITY = 0.60
MAX_HIGH_MISMATCHES = 3
MEDIUM_CONTACT_IDENTITY = 0.40
DIVERGENT_CONTACT_IDENTITY = 0.40
CHARGE_INVERSION_MIN_DELTA = 2

HIGH_MEAN_CA_DISTANCE = 4.0
MEDIUM_MEAN_CA_DISTANCE = 6.0
HIGH_CONTACTS_WITHIN_THRESHOLD = 14
MEDIUM_CONTACTS_WITHIN_THRESHOLD = 10
POOR_SUPERPOSITION_MEAN_CA = 5.0

HIGH_CONTACT_DRMSD = 2.0
MEDIUM_CONTACT_DRMSD = 4.0
POOR_CONTACT_DRMSD = 6.0


@dataclass(frozen=True, slots=True)
class MappingConfidenceInputs:
    """Inputs for structural mapping confidence assignment."""

    contact_mapping_fraction: float
    mean_plddt_pocket: float | None
    mapped_pocket_residues: int
    mean_contact_ca_distance: float | None
    n_contacts_within_threshold: int
    n_superposition_pairs: int
    sbd_superposition_rmsd: float | None
    contact_drmsd: float | None
    mapping_mode: MappingMode
    sbd_alignment_coverage: float


@dataclass(frozen=True, slots=True)
class QualityFlagInputs:
    """Inputs for pocket charge quality flag generation."""

    contact_mapping_fraction: float
    mean_plddt_pocket: float | None
    mapped_pocket_residues: int
    n_contact_mismatches: int
    n_low_plddt_excluded: int
    mapping_mode: MappingMode
    sbd_alignment_coverage: float
    contact_sequence_identity: float
    mean_contact_ca_distance: float | None
    contact_drmsd: float | None
    sbd_superposition_rmsd: float | None
    mapping_confidence: ConfidenceTier
    conservation_score: ConfidenceTier
    delta_contact_net_charge: int | None


def tier_rank(tier: ConfidenceTier) -> int:
    """Numeric rank for --min-confidence filtering (higher is stricter)."""
    return {"high": 3, "medium": 2, "low": 1}[tier]


def _min_tier(a: ConfidenceTier, b: ConfidenceTier) -> ConfidenceTier:
    return a if tier_rank(a) <= tier_rank(b) else b


def _mapping_confidence_from_superposition(
    contact_mapping_fraction: float,
    mean_plddt_pocket: float | None,
    mapped_pocket_residues: int,
    mean_contact_ca_distance: float,
    n_contacts_within_threshold: int,
) -> ConfidenceTier:
    if (
        contact_mapping_fraction >= HIGH_CONTACT_MAPPING
        and mapped_pocket_residues >= MIN_MAPPED_POCKET_RESIDUES
        and mean_plddt_pocket is not None
        and mean_plddt_pocket >= MIN_MEAN_PLDDT
        and mean_contact_ca_distance <= HIGH_MEAN_CA_DISTANCE
        and n_contacts_within_threshold >= HIGH_CONTACTS_WITHIN_THRESHOLD
    ):
        return "high"
    if (
        contact_mapping_fraction >= MEDIUM_CONTACT_MAPPING
        and mean_contact_ca_distance <= MEDIUM_MEAN_CA_DISTANCE
        and n_contacts_within_threshold >= MEDIUM_CONTACTS_WITHIN_THRESHOLD
    ):
        return "medium"
    return "low"


def _mapping_confidence_from_drmsd(
    contact_mapping_fraction: float,
    mean_plddt_pocket: float | None,
    mapped_pocket_residues: int,
    contact_drmsd: float,
) -> ConfidenceTier:
    if (
        contact_mapping_fraction >= HIGH_CONTACT_MAPPING
        and mapped_pocket_residues >= MIN_MAPPED_POCKET_RESIDUES
        and mean_plddt_pocket is not None
        and mean_plddt_pocket >= MIN_MEAN_PLDDT
        and contact_drmsd <= HIGH_CONTACT_DRMSD
    ):
        return "high"
    if contact_mapping_fraction >= MEDIUM_CONTACT_MAPPING and contact_drmsd <= MEDIUM_CONTACT_DRMSD:
        return "medium"
    return "low"


def _mapping_confidence_from_alignment(
    contact_mapping_fraction: float,
    mean_plddt_pocket: float | None,
    mapped_pocket_residues: int,
    mapping_mode: MappingMode,
    sbd_alignment_coverage: float,
) -> ConfidenceTier:
    """Fallback when AlphaFold conformations prevent cross-structure superposition."""
    if (
        contact_mapping_fraction >= HIGH_CONTACT_MAPPING
        and mapped_pocket_residues >= MIN_MAPPED_POCKET_RESIDUES
        and mean_plddt_pocket is not None
        and mean_plddt_pocket >= MIN_MEAN_PLDDT
        and mapping_mode == "sbd_local"
        and sbd_alignment_coverage >= MIN_SBD_ALIGNMENT_COVERAGE
    ):
        return "high"
    if (
        contact_mapping_fraction >= MEDIUM_CONTACT_MAPPING
        and mapping_mode == "sbd_local"
        and sbd_alignment_coverage >= MIN_SBD_ALIGNMENT_COVERAGE
    ):
        return "medium"
    return "low"


def assign_mapping_confidence(inputs: MappingConfidenceInputs) -> ConfidenceTier:
    """Assign structural mapping confidence (independent of sequence conservation)."""
    if inputs.n_superposition_pairs < MIN_SBD_PAIRS_FOR_SUPERPOSITION:
        return _mapping_confidence_from_alignment(
            inputs.contact_mapping_fraction,
            inputs.mean_plddt_pocket,
            inputs.mapped_pocket_residues,
            inputs.mapping_mode,
            inputs.sbd_alignment_coverage,
        )

    superposition_reliable = (
        inputs.sbd_superposition_rmsd is not None and inputs.sbd_superposition_rmsd < RELIABLE_SUPERPOSITION_RMSD
    )
    if superposition_reliable and inputs.mean_contact_ca_distance is not None:
        return _mapping_confidence_from_superposition(
            inputs.contact_mapping_fraction,
            inputs.mean_plddt_pocket,
            inputs.mapped_pocket_residues,
            inputs.mean_contact_ca_distance,
            inputs.n_contacts_within_threshold,
        )

    if inputs.contact_drmsd is not None:
        tier = _mapping_confidence_from_drmsd(
            inputs.contact_mapping_fraction,
            inputs.mean_plddt_pocket,
            inputs.mapped_pocket_residues,
            inputs.contact_drmsd,
        )
        if tier != "low":
            return tier

    return _mapping_confidence_from_alignment(
        inputs.contact_mapping_fraction,
        inputs.mean_plddt_pocket,
        inputs.mapped_pocket_residues,
        inputs.mapping_mode,
        inputs.sbd_alignment_coverage,
    )


def assign_conservation_score(
    contact_sequence_identity: float,
    n_contact_mismatches: int,
) -> ConfidenceTier:
    """Assign sequence conservation at contact sites."""
    if contact_sequence_identity >= HIGH_CONTACT_IDENTITY and n_contact_mismatches <= MAX_HIGH_MISMATCHES:
        return "high"
    if contact_sequence_identity >= MEDIUM_CONTACT_IDENTITY:
        return "medium"
    return "low"


def assign_confidence_tier(
    mapping_confidence: ConfidenceTier,
    conservation_score: ConfidenceTier,
) -> ConfidenceTier:
    """Combined conservative tier for --min-confidence filtering."""
    return _min_tier(mapping_confidence, conservation_score)


def _alignment_quality_flags(inputs: QualityFlagInputs) -> list[str]:
    flags: list[str] = []
    if inputs.mapping_mode == "full_length":
        flags.append("full_length_alignment")
    if inputs.sbd_alignment_coverage > 0 and inputs.sbd_alignment_coverage < MIN_SBD_ALIGNMENT_COVERAGE:
        flags.append("low_sbd_coverage")
    if inputs.contact_mapping_fraction < HIGH_CONTACT_MAPPING:
        flags.append("low_contact_mapping")
    if inputs.contact_sequence_identity < DIVERGENT_CONTACT_IDENTITY:
        flags.append("divergent_contact_sequences")
    if inputs.contact_sequence_identity < HIGH_CONTACT_IDENTITY:
        flags.append("low_contact_identity")
    if inputs.n_contact_mismatches > 0:
        flags.append("contact_mismatches")
    if inputs.n_contact_mismatches > MAX_HIGH_MISMATCHES:
        flags.append("many_contact_mismatches")
    return flags


def _structure_quality_flags(inputs: QualityFlagInputs) -> list[str]:
    flags: list[str] = []
    if inputs.sbd_superposition_rmsd is not None and inputs.sbd_superposition_rmsd >= RELIABLE_SUPERPOSITION_RMSD:
        flags.append("unreliable_superposition")
    if inputs.mean_contact_ca_distance is not None and inputs.mean_contact_ca_distance > POOR_SUPERPOSITION_MEAN_CA:
        flags.append("poor_structural_superposition")
    if inputs.contact_drmsd is not None and inputs.contact_drmsd > POOR_CONTACT_DRMSD:
        flags.append("poor_contact_geometry")
    return flags


def _pocket_quality_flags(inputs: QualityFlagInputs) -> list[str]:
    flags: list[str] = []
    if inputs.mapped_pocket_residues < MIN_MAPPED_POCKET_RESIDUES:
        flags.append("few_pocket_residues")
    if inputs.mean_plddt_pocket is not None and inputs.mean_plddt_pocket < MIN_MEAN_PLDDT:
        flags.append("low_plddt")
    elif inputs.mean_plddt_pocket is None:
        flags.append("plddt_unavailable")
    if inputs.n_low_plddt_excluded > 0:
        flags.append("plddt_filtered")
    return flags


def _charge_candidate_flags(inputs: QualityFlagInputs) -> list[str]:
    if (
        inputs.mapping_confidence == "high"
        and inputs.conservation_score == "low"
        and inputs.delta_contact_net_charge is not None
        and abs(inputs.delta_contact_net_charge) >= CHARGE_INVERSION_MIN_DELTA
    ):
        return ["charge_inversion_candidate"]
    return []


def build_quality_flags(inputs: QualityFlagInputs) -> list[str]:
    """Return structured quality flags for interpretability."""
    flags: list[str] = []
    flags.extend(_alignment_quality_flags(inputs))
    flags.extend(_structure_quality_flags(inputs))
    flags.extend(_pocket_quality_flags(inputs))
    flags.extend(_charge_candidate_flags(inputs))
    return flags
