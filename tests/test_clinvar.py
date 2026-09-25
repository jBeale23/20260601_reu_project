"""Tests for ClinVar variant evidence.

The curated-disease route reached ten proteins, too few to test anything. ClinVar offers
hundreds of genes' worth of variant counts instead - but only if the counts mean what they
appear to. Most of these tests exist to stop them meaning something else.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from data_fetching.fetch_clinvar import (
    GENE_SPECIFIC_TYPES,
    MIN_CLASSIFIED_VARIANTS,
    PATHOGENIC_TERMS,
    VariantCounts,
    VariantStore,
    _term,
    load_variant_store,
    write_variant_store,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_copy_number_events_are_excluded_by_default() -> None:
    """A correctness fix, not a refinement.

    Large copy-number events span hundreds of genes and are recorded as pathogenic for each.
    Unfiltered, DNAJA1 - which causes no known disease - showed 71 pathogenic variants, 67
    of them copy-number events and one an actual SNV.
    """
    term = _term("DNAJB6", PATHOGENIC_TERMS)
    assert "[Type]" in term
    assert "single nucleotide variant" in term
    assert "copy number" not in term.lower()


def test_the_unfiltered_query_is_still_reachable() -> None:
    """The excluded count is reported, so the filter's size stays visible."""
    assert "[Type]" not in _term("DNAJB6", PATHOGENIC_TERMS, gene_specific_only=False)


def test_a_gene_only_query_still_restricts_variant_type() -> None:
    """Otherwise the total and the classified counts would be on different footings."""
    assert "[Type]" in _term("DNAJB6")


def test_the_pathogenic_fraction_uses_only_confident_calls() -> None:
    """Variants of uncertain significance are the majority and say nothing either way."""
    counts = VariantCounts(gene="G", n_pathogenic=20, n_benign=80, n_uncertain=500)
    assert counts.n_classified == 100
    assert counts.pathogenic_fraction == pytest.approx(0.2)


def test_a_gene_with_nothing_classified_reports_zero_not_a_crash() -> None:
    """Zero here means "no evidence", which is not the same as "no pathogenic variants"."""
    counts = VariantCounts(gene="G", n_total=40, n_uncertain=40)
    assert counts.pathogenic_fraction == 0.0
    assert counts.is_informative is False


def test_a_handful_of_variants_is_not_informative() -> None:
    """Below a few classified variants the fraction swings on one submission."""
    assert VariantCounts(gene="G", n_pathogenic=1, n_benign=1).is_informative is False
    assert VariantCounts(gene="G", n_pathogenic=3, n_benign=3).is_informative is True
    assert MIN_CLASSIFIED_VARIANTS >= 5


def test_the_store_round_trips(tmp_path: Path) -> None:
    """A resumed fetch must rebuild exactly what it wrote."""
    path = tmp_path / "clinvar.json"
    store = VariantStore()
    store.counts["DNAJB6"] = VariantCounts(
        gene="DNAJB6",
        n_total=628,
        n_pathogenic=52,
        n_benign=319,
        n_uncertain=200,
        n_conflicting=5,
        n_excluded_copy_number=74,
    )
    store.failures["BAD"] = "ClientError"
    write_variant_store(store, path)
    back = load_variant_store(path)
    assert back.counts["DNAJB6"].n_pathogenic == 52
    assert back.counts["DNAJB6"].n_excluded_copy_number == 74
    assert back.failures == {"BAD": "ClientError"}


def test_writing_is_atomic(tmp_path: Path) -> None:
    """A kill mid-write must leave the previous store readable, not truncated."""
    path = tmp_path / "clinvar.json"
    write_variant_store(VariantStore(), path)
    assert json.loads(path.read_text())["counts"] == {}
    assert not list(tmp_path.glob("*.part"))


def test_every_gene_specific_type_is_a_real_clinvar_type() -> None:
    """A typo here would silently return zero for every gene."""
    assert set(GENE_SPECIFIC_TYPES) == {
        "single nucleotide variant",
        "deletion",
        "insertion",
        "indel",
        "duplication",
    }
