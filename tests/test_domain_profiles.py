"""Tests for layout-based class A/B/C assignment and novel-category detection."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import pytest

from domain_layout.constants import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_UNKNOWN,
    GF_RICH_FRACTION_THRESHOLD,
    NOVELTY_THRESHOLD,
    SUBCLASS_A_CANONICAL,
    SUBCLASS_A_NO_CTD,
    SUBCLASS_B_CANONICAL,
    SUBCLASS_C_J_ONLY,
    SUBCLASS_C_MEMBRANE,
    SUBCLASS_C_MULTIDOMAIN,
    SUBCLASS_C_SECRETORY,
    SUBCLASS_C_UNASSIGNED,
    SUBCLASS_UNKNOWN,
)
from domain_layout.disorder import BACKEND_FOLDINDEX, predict_disorder
from domain_layout.profiles import (
    NOVEL_SHARK_SIMILARITY,
    NOVELTY_SCORE_DECIMALS,
    NOVELTY_WEIGHT_J_DOMAIN_PLACEMENT,
    NOVELTY_WEIGHT_NO_CANONICAL_PARTNER,
    NOVELTY_WEIGHT_NO_HPD,
    NOVELTY_WEIGHT_SHARK_DISSIMILAR,
    NOVELTY_WEIGHT_UNKNOWN_PARTNER,
    LayoutEvidence,
    build_evidence,
    classify_layout,
    gf_fraction,
    max_gf_window_fraction,
    novelty_score,
    region_idr_fraction,
)
from domain_layout.regions import segment_protein
from domain_layout.shark import BACKEND_KMER, SharkMatch
from tests.conftest import make_entry, make_record

if TYPE_CHECKING:
    from domain_layout.records import ProteinDomainRecord


def _evidence(**overrides: object) -> LayoutEvidence:
    fields: dict[str, object] = {
        "accession": "TEST",
        "protein_length": 300,
        "has_j_domain": True,
        "has_hpd": True,
        "has_dnaj_c": False,
        "has_zinc_finger_like": False,
        "has_gf_rich_region": False,
        "has_transmembrane": False,
        "has_signal_peptide": False,
        "j_domain_position": "n_terminal",
        "n_structured_domains": 1,
        "unknown_partner_families": 0,
        "idr_fraction": 0.2,
        "domain_family_layout": "j_domain",
        "structured_families": ("j_domain",),
    }
    fields.update(overrides)
    return LayoutEvidence(**fields)  # type: ignore[arg-type]


def test_gf_fraction() -> None:
    """G/F fraction counts glycine and phenylalanine only."""
    assert gf_fraction("GGFF") == pytest.approx(1.0)
    assert gf_fraction("GGAA") == pytest.approx(0.5)
    assert gf_fraction("") == 0.0


def test_class_a_requires_zinc_finger() -> None:
    """The zinc-finger-like region is the class A discriminator."""
    canonical = classify_layout(
        _evidence(has_zinc_finger_like=True, has_dnaj_c=True, n_structured_domains=3),
    )
    assert canonical.predicted_class == CLASS_A
    assert canonical.predicted_subclass == SUBCLASS_A_CANONICAL
    assert canonical.class_confidence == "high"

    without_ctd = classify_layout(_evidence(has_zinc_finger_like=True, n_structured_domains=2))
    assert without_ctd.predicted_class == CLASS_A
    assert without_ctd.predicted_subclass == SUBCLASS_A_NO_CTD


def test_class_b_from_c_terminal_domain_without_zinc_finger() -> None:
    """J-domain plus the C-terminal substrate-binding domain is class B."""
    result = classify_layout(_evidence(has_dnaj_c=True, n_structured_domains=2))
    assert result.predicted_class == CLASS_B
    assert result.predicted_subclass == SUBCLASS_B_CANONICAL


def test_class_c_subclasses() -> None:
    """Class C is subdivided by localization and domain layout."""
    membrane = classify_layout(_evidence(has_transmembrane=True))
    assert membrane.predicted_class == CLASS_C
    assert membrane.predicted_subclass == SUBCLASS_C_MEMBRANE

    j_only = classify_layout(_evidence())
    assert j_only.predicted_subclass == SUBCLASS_C_J_ONLY

    multi = classify_layout(
        _evidence(n_structured_domains=3, structured_families=("j_domain", "other_domain")),
    )
    assert multi.predicted_subclass == SUBCLASS_C_MULTIDOMAIN


def test_unknown_without_j_domain() -> None:
    """No J-domain means no JDP class and no novelty score."""
    result = classify_layout(_evidence(has_j_domain=False, has_hpd=False))
    assert result.predicted_class == CLASS_UNKNOWN
    assert result.predicted_subclass == SUBCLASS_UNKNOWN
    assert result.novelty_score == 0.0
    assert result.novel_class_candidate is False


def test_novelty_terms_accumulate() -> None:
    """Each novelty term contributes its documented weight and tag."""
    evidence = _evidence(
        has_hpd=False,
        j_domain_position="internal",
        n_structured_domains=2,
        unknown_partner_families=1,
        structured_families=("j_domain", "other_domain"),
    )
    score, tags = novelty_score(evidence, None)
    assert score == pytest.approx(0.25 + 0.15 + 0.20 + 0.15)
    assert "no_hpd_in_j_domain" in tags
    assert "j_domain_internal" in tags
    assert "no_canonical_partner_domain" in tags
    assert "unannotated_partner_domain" in tags
    assert "no_reference_similarity" in tags


def test_novelty_adds_shark_term_only_when_dissimilar() -> None:
    """A close reference match suppresses the SHARK novelty term."""
    evidence = _evidence(has_hpd=False)
    dissimilar = SharkMatch(reference_id="ref", score=NOVEL_SHARK_SIMILARITY - 0.1, backend=BACKEND_KMER)
    similar = SharkMatch(reference_id="ref", score=NOVEL_SHARK_SIMILARITY + 0.1, backend=BACKEND_KMER)

    dissimilar_score, dissimilar_tags = novelty_score(evidence, dissimilar)
    similar_score, similar_tags = novelty_score(evidence, similar)

    assert dissimilar_score > similar_score
    assert "shark_dissimilar_to_reference_classes" in dissimilar_tags
    assert "shark_dissimilar_to_reference_classes" not in similar_tags


def test_novel_candidate_requires_threshold() -> None:
    """A protein crosses into the novel category only above the novelty threshold."""
    evidence = _evidence(
        has_hpd=False,
        j_domain_position="c_terminal",
        unknown_partner_families=1,
        n_structured_domains=2,
        structured_families=("j_domain", "other_domain"),
    )
    result = classify_layout(
        evidence,
        shark_match=SharkMatch(reference_id="ref", score=0.05, backend=BACKEND_KMER),
    )
    assert result.novel_class_candidate is True
    assert result.novelty_score == pytest.approx(1.0)
    assert "novel_class_candidate" in result.evidence_tags
    # The structural call survives: novelty is a separate axis, not a replacement label.
    assert result.predicted_subclass == SUBCLASS_C_MULTIDOMAIN

    typical = classify_layout(_evidence(has_dnaj_c=True))
    assert typical.novel_class_candidate is False


def test_confidence_tiers() -> None:
    """Confidence combines HPD evidence with canonical partner domains."""
    assert classify_layout(_evidence(has_dnaj_c=True)).class_confidence == "high"
    assert classify_layout(_evidence(has_hpd=False, has_dnaj_c=True)).class_confidence == "medium"
    assert classify_layout(_evidence(has_hpd=True)).class_confidence == "medium"
    assert classify_layout(_evidence(has_hpd=False)).class_confidence == "low"


def test_build_evidence_from_real_dnaj_layout(dnaj_record: ProteinDomainRecord) -> None:
    """E. coli DnaJ yields class A evidence straight from its domain record."""
    disorder = predict_disorder(dnaj_record.sequence, backend=BACKEND_FOLDINDEX)
    regions = segment_protein(dnaj_record, disorder)
    evidence = build_evidence(dnaj_record, regions)

    assert evidence.has_j_domain is True
    assert evidence.has_hpd is True
    assert evidence.has_zinc_finger_like is True
    assert evidence.has_dnaj_c is True
    assert evidence.j_domain_position == "n_terminal"
    assert evidence.domain_family_layout.startswith("j_domain>dnaj_c>zinc_finger_like")

    result = classify_layout(evidence)
    assert result.predicted_class == CLASS_A
    assert result.novel_class_candidate is False


def test_gf_rich_idr_counts_as_canonical_partner() -> None:
    """A glycine/phenylalanine-rich IDR after the J-domain is class B evidence."""
    sequence = "M" * 40 + "GGFGGGFGGGFGGGFGGGFGGGFGGGFGGGFGGSGSG" + "K" * 40
    record = make_record("TEST", sequence, entries=(make_entry("PF00226", 1, 40),))
    disorder = predict_disorder(sequence, backend=BACKEND_FOLDINDEX)
    regions = segment_protein(record, disorder)
    evidence = build_evidence(record, regions)
    assert evidence.has_gf_rich_region is True
    assert classify_layout(evidence).predicted_class == CLASS_B


def test_region_idr_fraction_excludes_annotated_domains(dnaj_record: ProteinDomainRecord) -> None:
    """IDR fraction is computed after domains are carved out, so it stays conservative."""
    disorder = predict_disorder(dnaj_record.sequence, backend=BACKEND_FOLDINDEX)
    regions = segment_protein(dnaj_record, disorder)
    fraction = region_idr_fraction(regions, len(dnaj_record.sequence))
    assert 0.0 <= fraction < disorder.idr_fraction


def test_max_gf_window_fraction_finds_local_blocks() -> None:
    """A G/F-rich block is detected even when embedded in a long non-G/F region."""
    block = "GGFGGGFGGGFGGGFGGGFGGGFGGGFGGG"
    assert max_gf_window_fraction(block) == pytest.approx(1.0)
    assert max_gf_window_fraction("K" * 200 + block + "E" * 200) == pytest.approx(1.0)
    assert max_gf_window_fraction("K" * 200) == pytest.approx(0.0)
    assert max_gf_window_fraction("") == 0.0


def test_max_gf_window_fraction_short_sequences_use_whole_region() -> None:
    """Regions shorter than the window fall back to whole-region composition."""
    assert max_gf_window_fraction("GGFF", window=30) == pytest.approx(1.0)
    assert max_gf_window_fraction("GGAA", window=30) == pytest.approx(0.5)


def test_gf_detection_is_independent_of_region_boundaries() -> None:
    """The same G/F block is detected whether or not it is merged into a longer IDR.

    Whole-region composition would dilute the block below threshold once a disorder
    predictor merged it into a larger region, flipping a class B protein to class C.
    """
    block = "GGFGGGFGGGFGGGFGGGFGGGFGGGFGGG"
    isolated = block
    merged = block + "KEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKE" * 2

    assert max_gf_window_fraction(isolated) >= GF_RICH_FRACTION_THRESHOLD
    assert max_gf_window_fraction(merged) >= GF_RICH_FRACTION_THRESHOLD
    assert gf_fraction(merged) < GF_RICH_FRACTION_THRESHOLD


def test_j_domain_position_variants() -> None:
    """J-domain placement is reported as N-terminal, internal, or C-terminal."""
    sequence = "M" * 400
    entries_by_case = {
        "n_terminal": (make_entry("PF00226", 1, 70), make_entry("PF01556", 200, 380)),
        "internal": (make_entry("PF99999", 1, 120), make_entry("PF00226", 160, 230), make_entry("PF88888", 260, 380)),
        "c_terminal": (make_entry("PF01556", 1, 200), make_entry("PF00226", 330, 395)),
    }
    for expected, entries in entries_by_case.items():
        record = make_record("TEST", sequence, entries=entries)
        regions = segment_protein(record, predict_disorder(sequence, backend=BACKEND_FOLDINDEX))
        assert build_evidence(record, regions).j_domain_position == expected, expected


def test_j_domain_position_unknown_without_j_domain() -> None:
    """A protein with no J-domain region has an unknown J-domain position."""
    sequence = "M" * 200
    record = make_record("TEST", sequence, entries=(make_entry("PF01556", 10, 150),))
    regions = segment_protein(record, predict_disorder(sequence, backend=BACKEND_FOLDINDEX))
    evidence = build_evidence(record, regions)
    assert evidence.j_domain_position == "unknown"
    assert evidence.has_j_domain is False


def test_secretory_subclass_from_signal_peptide() -> None:
    """A signal peptide without a TM segment gives the secretory class C subclass."""
    result = classify_layout(_evidence(has_signal_peptide=True))
    assert result.predicted_class == CLASS_C
    assert result.predicted_subclass == SUBCLASS_C_SECRETORY
    assert "secretory_signal" in result.evidence_tags


def test_transmembrane_wins_over_signal_peptide() -> None:
    """A protein with both TM and signal evidence is tagged membrane-associated first."""
    result = classify_layout(_evidence(has_transmembrane=True, has_signal_peptide=True))
    assert result.predicted_subclass == SUBCLASS_C_MEMBRANE
    assert "membrane_associated" in result.evidence_tags
    assert "secretory_signal" in result.evidence_tags


def test_region_idr_fraction_handles_zero_length() -> None:
    """A zero-length protein reports no IDR fraction instead of dividing by zero."""
    assert region_idr_fraction([], 0) == 0.0
    assert region_idr_fraction([], 100) == 0.0


def test_idr_dominated_tag_applied_above_threshold() -> None:
    """Highly disordered proteins carry the idr_dominated tag."""
    assert "idr_dominated" in classify_layout(_evidence(idr_fraction=0.9)).evidence_tags
    assert "idr_dominated" not in classify_layout(_evidence(idr_fraction=0.1)).evidence_tags


def test_annotated_gf_rich_domain_is_canonical_partner_evidence() -> None:
    """A Pfam-annotated G/F-rich domain (PF09320) counts without composition scoring."""
    sequence = "M" * 40 + "AAAAKKKKEEEEAAAAKKKKEEEEAAAAKKKK" + "M" * 40
    record = make_record(
        "TEST",
        sequence,
        entries=(make_entry("PF00226", 1, 40), make_entry("PF09320", 41, 72)),
    )
    regions = segment_protein(record, predict_disorder(sequence, backend=BACKEND_FOLDINDEX))
    evidence = build_evidence(record, regions)

    assert evidence.has_gf_rich_region is True
    assert evidence.has_canonical_partner is True
    assert classify_layout(evidence).predicted_class == CLASS_B


def test_class_c_unassigned_is_the_documented_fallback() -> None:
    """A J-domain protein matching no other class C rule falls back to c_unassigned."""
    result = classify_layout(
        _evidence(n_structured_domains=1, structured_families=("other_domain",), idr_fraction=0.1),
    )
    assert result.predicted_class == CLASS_C
    assert result.predicted_subclass == SUBCLASS_C_UNASSIGNED


def test_novelty_decision_matches_the_reported_score() -> None:
    """The flag is decided on the same rounded number that appears in the output.

    Deciding on the raw sum would be fragile: none of the weights is exactly
    representable in binary floating point, and several combinations land exactly on the
    threshold, so a future weight change could silently unflag a whole category.
    """
    evidence = _evidence(
        has_hpd=True,
        j_domain_position="c_terminal",
        unknown_partner_families=1,
        n_structured_domains=2,
        structured_families=("j_domain", "other_domain"),
    )
    result = classify_layout(evidence)

    # placement (0.15) + no canonical partner (0.20) + unannotated partner (0.15) = 0.50
    assert result.novelty_score == pytest.approx(NOVELTY_THRESHOLD)
    assert result.novel_class_candidate is True
    assert (result.novelty_score >= NOVELTY_THRESHOLD) is result.novel_class_candidate


def test_every_novelty_weight_combination_is_decided_consistently() -> None:
    """No subset of weights may disagree with exact arithmetic at the threshold.

    Guards future weight edits: if a new combination summed to 0.49999999999999994 the
    proteins triggering it would stop being flagged for no scientific reason.
    """
    weights = {
        "no_hpd": NOVELTY_WEIGHT_NO_HPD,
        "placement": NOVELTY_WEIGHT_J_DOMAIN_PLACEMENT,
        "no_partner": NOVELTY_WEIGHT_NO_CANONICAL_PARTNER,
        "unannotated": NOVELTY_WEIGHT_UNKNOWN_PARTNER,
        "shark": NOVELTY_WEIGHT_SHARK_DISSIMILAR,
    }
    for size in range(1, len(weights) + 1):
        for combo in combinations(weights, size):
            accumulated = 0.0
            for name in combo:
                accumulated += weights[name]
            reported = round(accumulated, NOVELTY_SCORE_DECIMALS)
            exact = sum((Fraction(str(weights[name])) for name in combo), Fraction(0))
            assert (reported >= NOVELTY_THRESHOLD) is (exact >= Fraction(1, 2)), combo
