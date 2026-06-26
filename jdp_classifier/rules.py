"""Rule-based class A/B/C assignment and confidence tiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jdp_classifier.constants import (
    MIN_CLASS_A_DOMAINS,
    MIN_CLASS_B_DOMAINS,
    ConfidenceTier,
    PredictedClass,
)

if TYPE_CHECKING:
    from jdp_classifier.architecture import ArchitectureFeatures


def predict_class(features: ArchitectureFeatures) -> PredictedClass:
    """Assign rule-based JDP class from architecture features."""
    if not features.has_j_domain:
        return "unknown"

    class_a = features.n_domains >= MIN_CLASS_A_DOMAINS and (features.has_dnaj_c or features.has_zinc_finger_like)
    if class_a:
        return "A"

    class_b = features.n_domains >= MIN_CLASS_B_DOMAINS and features.has_gf_rich
    if class_b:
        return "B"

    return "C"


def assign_class_confidence(
    predicted_class: PredictedClass,
    features: ArchitectureFeatures,
    *,
    has_hpd: bool,
    hpd_confidence: str,
    sequence_available: bool,
) -> ConfidenceTier:
    """Assign confidence tier for the predicted class label."""
    if predicted_class == "unknown":
        return "low"

    if predicted_class == "A" and features.has_dnaj_c and features.n_domains >= MIN_CLASS_A_DOMAINS:
        return "high"

    if predicted_class == "B" and features.has_gf_rich and has_hpd and hpd_confidence == "high":
        return "high"

    if has_hpd and hpd_confidence == "high" and features.n_domains >= MIN_CLASS_B_DOMAINS:
        return "medium"

    tier: ConfidenceTier = "low"
    if sequence_available and has_hpd and predicted_class in {"B", "C"} and features.n_domains >= MIN_CLASS_B_DOMAINS:
        tier = "medium"
    return tier


def build_quality_flags(
    features: ArchitectureFeatures,
    *,
    has_hpd: bool,
    hpd_source: str,
    appears_in_architecture_count: int | None,
    sequence_available: bool,
) -> list[str]:
    """Return semicolon-joinable data-quality flags."""
    flags: list[str] = []
    if not has_hpd:
        flags.append("no_hpd")
    if hpd_source == "uniprot":
        flags.append("sequence_from_uniprot")
    if not sequence_available:
        flags.append("no_sequence")
    if appears_in_architecture_count is not None and appears_in_architecture_count > 1:
        flags.append("multi_architecture")
    if not features.has_j_domain:
        flags.append("no_j_domain_in_ida")
    return flags


def build_layout_tags(
    features: ArchitectureFeatures,
    predicted_class: PredictedClass,
    *,
    has_transmembrane: bool,
    has_signal_peptide: bool,
) -> list[str]:
    """Return semicolon-joinable biological layout tags."""
    tags: list[str] = []
    if features.n_domains == 1 and features.has_j_domain:
        tags.append("j_domain_only")
    if has_transmembrane:
        tags.append("membrane_associated")
    if has_signal_peptide:
        tags.append("secretory_signal")
    if predicted_class == "C" and features.n_domains >= MIN_CLASS_B_DOMAINS:
        tags.append("atypical_multi_domain")
    if features.has_gf_rich and predicted_class == "C":
        tags.append("gf_rich_atypical")
    return tags


def tier_rank(tier: ConfidenceTier) -> int:
    """Numeric rank for --min-confidence filtering."""
    return {"high": 3, "medium": 2, "low": 1}[tier]
