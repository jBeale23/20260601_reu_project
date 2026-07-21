"""Tests for scripts/chaperone_profiles.py."""

from __future__ import annotations

from scripts.chaperone_profiles import (
    ChaperonePresence,
    build_classification_tags,
    chaperone_system_membership,
    derive_chaperone_classification_fields,
    min_confidence_tier,
    unified_confidence_tier,
)


def test_chaperone_system_membership_labels() -> None:
    """Membership reflects DnaK/DnaJ fetch presence."""
    assert chaperone_system_membership(has_dnak=False, has_dnaj=False) == "none"
    assert chaperone_system_membership(has_dnak=True, has_dnaj=False) == "dnak"
    assert chaperone_system_membership(has_dnak=False, has_dnaj=True) == "dnaj"
    assert chaperone_system_membership(has_dnak=True, has_dnaj=True) == "dual"


def test_min_confidence_tier_picks_weakest() -> None:
    """Unified tier uses the minimum of available confidence labels."""
    assert min_confidence_tier("high", "medium") == "medium"
    assert min_confidence_tier("low", "high", "") == "low"
    assert min_confidence_tier("", "bogus") == ""


def test_unified_confidence_tier_dual_track() -> None:
    """When both JDP and pocket exist, tier is min across both tracks."""
    assert (
        unified_confidence_tier(
            jdp_class_confidence="high",
            pocket_confidence_tier="medium",
            pocket_mapping_confidence="high",
            has_jdp=True,
            has_pocket=True,
        )
        == "medium"
    )
    assert (
        unified_confidence_tier(
            jdp_class_confidence="",
            pocket_confidence_tier="high",
            pocket_mapping_confidence="high",
            has_jdp=False,
            has_pocket=True,
        )
        == "high"
    )


def test_build_classification_tags_includes_dual_and_inversion() -> None:
    """Tags summarize membership, class, and pocket inversion."""
    tags = build_classification_tags(
        chaperone_membership="dual",
        jdp_row={
            "jdp_predicted_class": "A",
            "jdp_class_confidence": "high",
            "jdp_quality_flags": "multi_architecture",
        },
        pocket_row={"confidence_tier": "high", "mapping_confidence": "medium"},
        charge_inversion_candidate=True,
        has_motif=True,
    )
    assert "dual_chaperone_homolog" in tags
    assert "jdp_class_a" in tags
    assert "pocket_charge_inversion" in tags
    assert "motif_features_present" in tags
    assert "jdp_multi_architecture" in tags


def test_derive_chaperone_classification_fields() -> None:
    """Derived field bundle matches expected membership and unified tier."""
    fields = derive_chaperone_classification_fields(
        presence=ChaperonePresence(
            has_dnak=True,
            has_dnaj=True,
            has_jdp=True,
            has_pocket=True,
            has_motif=False,
            charge_inversion_candidate=False,
        ),
        jdp_row={"jdp_predicted_class": "B", "jdp_class_confidence": "medium", "jdp_quality_flags": ""},
        pocket_row={"confidence_tier": "high", "mapping_confidence": "low"},
    )
    assert fields["chaperone_system_membership"] == "dual"
    assert fields["unified_confidence_tier"] == "low"
    assert "jdp_class_b" in fields["classification_tags"]
