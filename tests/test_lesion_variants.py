"""Tests for the test that decides whether the lesion idea is real.

The detector is validated only on lesions built by hand, which proves it finds what it was
built to find. This is the falsifiable part, so these tests are mostly about making sure it
can return a negative - a test that cannot fail is not evidence.
"""

from __future__ import annotations

import pytest

from data_fetching.fetch_clinvar import VariantPosition
from domain_layout.syntax_errors import CHARGE_ANOMALY, LOW_COMPLEXITY, GrammaticalLesion
from validation.lesion_variants import (
    MIN_VARIANTS_PER_CLASS,
    SIGNIFICANCE_LEVEL,
    Coincidence,
    coincidence_by_scope,
    coincidence_report,
    usable_variants,
)

_SEQ = "MVKETKFYDILGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKEISEAYEVLSDPEKREIYDQ" * 3


def _variant(position: int, reference: str, *, pathogenic: bool) -> VariantPosition:
    return VariantPosition(
        gene="G",
        position=position,
        reference=reference,
        alternate="A",
        is_pathogenic=pathogenic,
    )


def _lesion(start: int, end: int, error_type: str = LOW_COMPLEXITY) -> GrammaticalLesion:
    return GrammaticalLesion(
        start=start,
        end=end,
        error_type=error_type,
        z_score=3.0,
        surprisal_per_residue=1.0,
        subsequence="X",
        detail="test",
    )


# Isoform matching
# ----------------


def test_a_variant_whose_reference_residue_matches_is_used() -> None:
    """The isoform check and the mapping validation are the same step."""
    reference = _SEQ[9]
    assert usable_variants([_variant(10, reference, pathogenic=True)], _SEQ)


def test_a_variant_numbered_against_another_isoform_is_dropped() -> None:
    """Counting it would place a real variant at an arbitrary position."""
    wrong = "W" if _SEQ[9] != "W" else "K"
    assert usable_variants([_variant(10, wrong, pathogenic=True)], _SEQ) == []


def test_a_position_past_the_end_is_dropped() -> None:
    """A transcript longer than the protein held here indexes off the end."""
    assert usable_variants([_variant(99999, "M", pathogenic=True)], _SEQ) == []


# The test itself
# ---------------


def test_pathogenic_variants_concentrated_in_lesions_are_detected() -> None:
    """The positive case the whole analysis exists to find."""
    lesions = [_lesion(20, 40)]
    variants = [_variant(p, _SEQ[p - 1], pathogenic=True) for p in range(21, 39)]
    variants += [_variant(p, _SEQ[p - 1], pathogenic=False) for p in range(60, 78)]
    results = coincidence_by_scope({"P1": (_SEQ, variants, lesions)})
    overall = next(r for r in results if r.scope == "any_lesion")
    assert overall.pathogenic_rate == pytest.approx(1.0)
    assert overall.benign_rate == pytest.approx(0.0)
    assert overall.odds_ratio > 10
    assert overall.p_adjusted < SIGNIFICANCE_LEVEL


def test_no_association_returns_a_negative() -> None:
    """A test that cannot fail is not evidence.

    Pathogenic and benign variants distributed identically must give an odds ratio near one
    and a non-significant p-value.
    """
    lesions = [_lesion(20, 40)]
    variants = [
        _variant(position, _SEQ[position - 1], pathogenic=position % 2 == 0)
        for position in [*range(21, 39), *range(60, 78)]
    ]
    results = coincidence_by_scope({"P1": (_SEQ, variants, lesions)})
    overall = next(r for r in results if r.scope == "any_lesion")
    assert 0.3 < overall.odds_ratio < 3.0
    assert overall.p_adjusted > SIGNIFICANCE_LEVEL


def test_each_lesion_type_is_tested_separately() -> None:
    """Pooling could hide a real effect in one type behind noise from another."""
    lesions = [_lesion(20, 40, CHARGE_ANOMALY), _lesion(80, 100, LOW_COMPLEXITY)]
    variants = [_variant(p, _SEQ[p - 1], pathogenic=True) for p in range(21, 39)]
    variants += [_variant(p, _SEQ[p - 1], pathogenic=False) for p in range(81, 99)]
    scopes = {r.scope for r in coincidence_by_scope({"P1": (_SEQ, variants, lesions)})}
    assert CHARGE_ANOMALY in scopes
    assert "any_lesion" in scopes


def test_too_few_variants_are_not_tested() -> None:
    """A 2x2 table over three variants is not a measurement."""
    lesions = [_lesion(20, 40)]
    variants = [_variant(21, _SEQ[20], pathogenic=True), _variant(60, _SEQ[59], pathogenic=False)]
    assert coincidence_by_scope({"P1": (_SEQ, variants, lesions)}) == []
    assert MIN_VARIANTS_PER_CLASS >= 5


def test_a_zero_cell_gives_a_finite_odds_ratio() -> None:
    """An infinity would dominate any ranking and hide the real effects."""
    item = Coincidence(
        scope="any_lesion", n_pathogenic_in=10, n_pathogenic_out=0, n_benign_in=0, n_benign_out=10, p_value=0.0
    )
    assert item.odds_ratio < float("inf")
    assert item.odds_ratio > 1


def test_the_report_states_why_benign_is_the_control() -> None:
    """The choice of control is the methodological claim; it belongs in the output."""
    lesions = [_lesion(20, 40)]
    variants = [_variant(p, _SEQ[p - 1], pathogenic=p % 2 == 0) for p in range(21, 45)]
    report = coincidence_report({"P1": (_SEQ, variants, lesions)})
    assert "shuffled positions" in str(report["interpretation"])
    assert report["n_variants_matched_to_sequence"] > 0


def test_isoform_mismatches_are_counted_not_hidden() -> None:
    """How many variants could not be placed is part of the result."""
    lesions = [_lesion(20, 40)]
    bad = [_variant(10, "W" if _SEQ[9] != "W" else "K", pathogenic=True) for _ in range(3)]
    good = [_variant(p, _SEQ[p - 1], pathogenic=p % 2 == 0) for p in range(21, 45)]
    report = coincidence_report({"P1": (_SEQ, bad + good, lesions)})
    assert report["n_variants_dropped_isoform_mismatch"] == 3
