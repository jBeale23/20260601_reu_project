"""Derive unified DnaJ/DnaK chaperone classification columns for merged feature tables."""

from __future__ import annotations

from dataclasses import dataclass

from jdp_classifier.rules import tier_rank

ConfidenceTier = str


@dataclass(frozen=True, slots=True)
class ChaperonePresence:
    """Which chaperone tracks contributed data for one merged accession."""

    has_dnak: bool
    has_dnaj: bool
    has_jdp: bool
    has_pocket: bool
    has_motif: bool
    charge_inversion_candidate: bool


def chaperone_system_membership(*, has_dnak: bool, has_dnaj: bool) -> str:
    """Label whether an accession appears in DnaK fetch, DnaJ fetch, or both."""
    if has_dnak and has_dnaj:
        return "dual"
    if has_dnaj:
        return "dnaj"
    if has_dnak:
        return "dnak"
    return "none"


def min_confidence_tier(*tiers: str) -> str:
    """Return the weakest non-empty confidence tier (high > medium > low)."""
    present = [tier for tier in tiers if tier in {"high", "medium", "low"}]
    if not present:
        return ""
    return min(present, key=tier_rank)


def unified_confidence_tier(
    *,
    jdp_class_confidence: str,
    pocket_confidence_tier: str,
    pocket_mapping_confidence: str,
    has_jdp: bool,
    has_pocket: bool,
) -> str:
    """Combine JDP and pocket confidence when both are available."""
    if has_jdp and has_pocket:
        pocket_tier = min_confidence_tier(pocket_confidence_tier, pocket_mapping_confidence)
        return min_confidence_tier(jdp_class_confidence, pocket_tier)
    if has_jdp:
        return jdp_class_confidence
    if has_pocket:
        return min_confidence_tier(pocket_confidence_tier, pocket_mapping_confidence)
    return ""


def build_classification_tags(
    *,
    chaperone_membership: str,
    jdp_row: dict[str, str],
    pocket_row: dict[str, str],
    charge_inversion_candidate: bool,
    has_motif: bool,
) -> str:
    """Semicolon-separated tags for filtering unified DnaJ/DnaK classification rows."""
    jdp_predicted_class = jdp_row.get("jdp_predicted_class", "")
    jdp_class_confidence = jdp_row.get("jdp_class_confidence", "")
    jdp_quality_flags = jdp_row.get("jdp_quality_flags", "")
    pocket_confidence_tier = pocket_row.get("confidence_tier", "")
    pocket_mapping_confidence = pocket_row.get("mapping_confidence", "")

    tags: list[str] = []
    if chaperone_membership == "dual":
        tags.append("dual_chaperone_homolog")
    elif chaperone_membership in {"dnaj", "dnak"}:
        tags.append(f"{chaperone_membership}_homolog")

    if jdp_predicted_class and jdp_predicted_class != "unknown":
        tags.append(f"jdp_class_{jdp_predicted_class.lower()}")
    if jdp_class_confidence in {"high", "medium", "low"}:
        tags.append(f"jdp_confidence_{jdp_class_confidence}")

    pocket_tier = min_confidence_tier(pocket_confidence_tier, pocket_mapping_confidence)
    if pocket_tier:
        tags.append(f"pocket_confidence_{pocket_tier}")
    if charge_inversion_candidate:
        tags.append("pocket_charge_inversion")
    if has_motif:
        tags.append("motif_features_present")
    if "no_hpd" in jdp_quality_flags.split(";"):
        tags.append("jdp_no_hpd")
    if "multi_architecture" in jdp_quality_flags.split(";"):
        tags.append("jdp_multi_architecture")

    return ";".join(tags)


def derive_chaperone_classification_fields(
    *,
    presence: ChaperonePresence,
    jdp_row: dict[str, str],
    pocket_row: dict[str, str],
) -> dict[str, str]:
    """Return derived columns linking DnaJ classifier output with DnaK pocket metrics."""
    membership = chaperone_system_membership(
        has_dnak=presence.has_dnak,
        has_dnaj=presence.has_dnaj,
    )
    unified = unified_confidence_tier(
        jdp_class_confidence=jdp_row.get("jdp_class_confidence", ""),
        pocket_confidence_tier=pocket_row.get("confidence_tier", ""),
        pocket_mapping_confidence=pocket_row.get("mapping_confidence", ""),
        has_jdp=presence.has_jdp,
        has_pocket=presence.has_pocket,
    )
    tags = build_classification_tags(
        chaperone_membership=membership,
        jdp_row=jdp_row,
        pocket_row=pocket_row,
        charge_inversion_candidate=presence.charge_inversion_candidate,
        has_motif=presence.has_motif,
    )
    return {
        "chaperone_system_membership": membership,
        "unified_confidence_tier": unified,
        "classification_tags": tags,
    }
