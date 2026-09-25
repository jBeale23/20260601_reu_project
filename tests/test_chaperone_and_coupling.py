"""Tests for chaperone plausibility, interaction evidence, and sequence-structure coupling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from data_fetching.fetch_biogrid import biogrid_available, record_from_interactions
from validation.chaperone_evidence import (
    MIN_MEMBERS_TO_JUDGE,
    SUSPECT_HPD_RATE,
    architecture_evidence,
    chaperone_report,
)
from validation.coupling import CoupledPair, classify_pairs, coupling_report, regime_by_group
from validation.coupling_cli import load_function_terms


class _Evidence:
    def __init__(self, *, architecture: str, hpd: bool, families: tuple[str, ...] = ()) -> None:
        self.domain_family_layout = architecture
        self.has_j_domain = True
        self.has_hpd = hpd
        self.structured_families = families


class _Layout:
    def __init__(self, **kwargs: object) -> None:
        self.evidence = _Evidence(**kwargs)


class _Hit:
    def __init__(self, query: str, target: str, identity: float, tm: float) -> None:
        self.query = query
        self.target = target
        self.sequence_identity = identity
        self.tm_score = tm


def test_hpd_rate_is_computed_per_architecture() -> None:
    """HPD is what makes a J-domain functional, so its rate is the chaperone test."""
    layouts = [_Layout(architecture="j>c", hpd=i < 18) for i in range(20)]
    evidence = architecture_evidence(layouts)["j>c"]
    assert evidence.n_proteins == 20
    assert evidence.hpd_rate == pytest.approx(0.9)
    assert not evidence.is_suspect


def test_an_architecture_missing_hpd_is_flagged() -> None:
    """A J-like domain folds the same way and cannot activate Hsp70."""
    layouts = [_Layout(architecture="odd", hpd=i < 5) for i in range(25)]
    evidence = architecture_evidence(layouts)["odd"]
    assert evidence.hpd_rate < SUSPECT_HPD_RATE
    assert evidence.is_suspect


def test_a_tiny_architecture_is_not_judged() -> None:
    """With three members, one missing motif looks like a 33% failure."""
    layouts = [_Layout(architecture="rare", hpd=False) for _ in range(3)]
    evidence = architecture_evidence(layouts)["rare"]
    assert not evidence.is_judgeable
    assert not evidence.is_suspect
    assert MIN_MEMBERS_TO_JUDGE > 3


def test_report_bands_hpd_rate_by_how_common_an_architecture_is() -> None:
    """The concern is specifically that the rare tail differs from the head.

    Measured on the full set it does not: 96.9% in the five commonest architectures
    against 95.7% beyond rank 50.
    """
    layouts = [_Layout(architecture=f"arch{rank:03d}", hpd=True) for rank in range(1, 60) for _ in range(100 - rank)]
    report = chaperone_report(layouts)

    bands = report["hpd_rate_by_architecture_rank"]
    assert "rank_1_5" in bands
    assert "rank_51_plus" in bands
    for band in bands.values():
        assert band["hpd_rate"] == pytest.approx(1.0)
    assert report["n_suspect_architectures"] == 0


def test_non_chaperone_hints_are_counted_not_excluded() -> None:
    """A co-chaperone can legitimately carry a DNA-binding domain.

    Whether such a protein belongs is a judgement about scope, not a fact about the
    sequence, so it is reported rather than dropped.
    """
    layouts = [_Layout(architecture="j>myb", hpd=True, families=("j_domain", "myb_like")) for _ in range(30)]
    evidence = architecture_evidence(layouts)["j>myb"]
    assert evidence.n_with_non_chaperone_hint == 30
    assert not evidence.is_suspect  # HPD is intact, so it is not excluded


def test_interaction_records_split_physical_from_genetic() -> None:
    """Genetic interactions reach relationships no physical assay would show."""
    payload = {
        "1": {"OFFICIAL_SYMBOL_A": "YDJ1", "OFFICIAL_SYMBOL_B": "SSA1", "EXPERIMENTAL_SYSTEM_TYPE": "physical"},
        "2": {"OFFICIAL_SYMBOL_A": "YDJ1", "OFFICIAL_SYMBOL_B": "HSP104", "EXPERIMENTAL_SYSTEM_TYPE": "genetic"},
    }
    record = record_from_interactions("YDJ1", payload)
    assert record.physical_partners == ("SSA1",)
    assert record.genetic_partners == ("HSP104",)
    assert record.n_partners == 2


def test_an_hsp70_partner_is_recognised_across_organisms() -> None:
    """A co-chaperone that never partners an Hsp70 is doing something else."""
    for partner in ("SSA1", "HSPA8", "DNAK", "KAR2"):
        payload = {
            "1": {"OFFICIAL_SYMBOL_A": "Q", "OFFICIAL_SYMBOL_B": partner, "EXPERIMENTAL_SYSTEM_TYPE": "physical"}
        }
        assert record_from_interactions("Q", payload).has_hsp70_partner, partner

    unrelated = {"1": {"OFFICIAL_SYMBOL_A": "Q", "OFFICIAL_SYMBOL_B": "ACT1", "EXPERIMENTAL_SYSTEM_TYPE": "physical"}}
    assert not record_from_interactions("Q", unrelated).has_hsp70_partner


def test_biogrid_reports_itself_unavailable_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key means the pipeline continues without interaction evidence, not that it fails."""
    monkeypatch.delenv("BIOGRID_ACCESS_KEY", raising=False)
    assert not biogrid_available()


def test_the_four_sequence_structure_regimes() -> None:
    """Only two of the four carry information about which changes matter."""
    both_conserved = CoupledPair("a", "b", 0.80, 0.90)
    permissive = CoupledPair("a", "b", 0.12, 0.88)
    decisive = CoupledPair("a", "b", 0.75, 0.55)
    both_diverged = CoupledPair("a", "b", 0.10, 0.55)

    assert both_conserved.regime == "both_conserved"
    assert permissive.regime == "sequence_diverged_structure_conserved"
    assert decisive.regime == "sequence_conserved_structure_diverged"
    assert both_diverged.regime == "both_diverged"

    assert permissive.is_informative
    assert decisive.is_informative
    assert not both_conserved.is_informative
    assert not both_diverged.is_informative


def test_pairs_below_the_fold_threshold_are_excluded() -> None:
    """Two proteins that do not share a fold cannot say anything about coupling."""
    hits = [_Hit("a", "b", 0.5, 0.30), _Hit("a", "c", 0.5, 0.80)]
    pairs = classify_pairs(hits)
    assert [pair.target for pair in pairs] == ["c"]


def test_shared_function_is_reported_only_when_both_sides_are_annotated() -> None:
    """Absent annotation must not be read as absent shared function."""
    hits = [_Hit("a", "b", 0.5, 0.80), _Hit("a", "c", 0.5, 0.80)]
    pairs = classify_pairs(hits, function_terms={"a": ["folding"], "b": ["folding"], "c": []})
    by_target = {pair.target: pair for pair in pairs}
    assert by_target["b"].shared_function is True
    assert by_target["c"].shared_function is None


def test_coupling_report_summarises_each_regime() -> None:
    """The report must name how often each regime occurs, not just list pairs."""
    hits = [_Hit("a", f"t{i}", 0.10, 0.85) for i in range(6)] + [_Hit("a", f"u{i}", 0.75, 0.55) for i in range(3)]
    report = coupling_report(classify_pairs(hits))

    assert report["n_pairs"] == 9
    regimes = report["regimes"]
    assert regimes["sequence_diverged_structure_conserved"]["n_pairs"] == 6
    assert regimes["sequence_conserved_structure_diverged"]["n_pairs"] == 3
    assert report["examples"]["sequence_conserved_structure_diverged"]


def test_regimes_can_be_compared_between_subpopulations() -> None:
    """A group under looser structural constraint should tolerate more divergence."""
    pairs = [
        CoupledPair("g1", "x", 0.10, 0.85),
        CoupledPair("g1", "y", 0.10, 0.85),
        CoupledPair("g2", "z", 0.75, 0.55),
    ]
    counts = regime_by_group(pairs, {"g1": "loose", "g2": "tight"})
    assert counts["loose"]["sequence_diverged_structure_conserved"] == 2
    assert counts["tight"]["sequence_conserved_structure_diverged"] == 1


def test_an_empty_pair_set_is_not_an_error() -> None:
    """No structural pairs means no coupling to report."""
    assert coupling_report([])["n_pairs"] == 0


def test_rows_are_attributed_only_to_the_gene_they_involve() -> None:
    """A batched response covers several genes; each row belongs to one of them.

    Without this check a row for one gene contributes a partner to every gene in the
    batch. A first run gave five different query proteins the same six partners.
    """
    payload = {
        "1": {"OFFICIAL_SYMBOL_A": "YDJ1", "OFFICIAL_SYMBOL_B": "SSA1", "EXPERIMENTAL_SYSTEM_TYPE": "physical"},
        "2": {"OFFICIAL_SYMBOL_A": "SIS1", "OFFICIAL_SYMBOL_B": "SSA2", "EXPERIMENTAL_SYSTEM_TYPE": "physical"},
    }
    assert record_from_interactions("YDJ1", payload).partners == ("SSA1",)
    assert record_from_interactions("SIS1", payload).partners == ("SSA2",)
    # A gene absent from the payload gets nothing, not everything.
    assert record_from_interactions("ZUO1", payload).partners == ()


def test_a_partner_is_found_whichever_column_the_query_occupies() -> None:
    """BioGRID puts the query in either column; matching only one loses half the data."""
    payload = {"1": {"OFFICIAL_SYMBOL_A": "RAD3", "OFFICIAL_SYMBOL_B": "YDJ1", "EXPERIMENTAL_SYSTEM_TYPE": "physical"}}
    assert record_from_interactions("YDJ1", payload).partners == ("RAD3",)


def test_self_interactions_are_excluded() -> None:
    """Real, but they say nothing about partner specificity."""
    payload = {"1": {"OFFICIAL_SYMBOL_A": "YDJ1", "OFFICIAL_SYMBOL_B": "YDJ1", "EXPERIMENTAL_SYSTEM_TYPE": "physical"}}
    assert record_from_interactions("YDJ1", payload).partners == ()


def test_an_empty_response_is_a_list_not_a_mapping() -> None:
    """BioGRID returns [] when nothing matches, and {} of rows otherwise."""
    assert record_from_interactions("X", []).n_partners == 0
    assert record_from_interactions("X", {}).n_partners == 0


# The coupling runner
# -------------------
# The analysis was implemented and tested long before anything could invoke it, and
# Foldseek was not installed, so it had never run. These pin the loader boundary.


def test_no_function_store_suppresses_the_shared_function_column(tmp_path: Path) -> None:
    """``None`` and an empty mapping mean different things and must not be conflated.

    ``None`` omits the column; an empty mapping would report every pair as *not* sharing
    function, which is a claim the data does not support.
    """
    assert load_function_terms(None) is None
    assert load_function_terms(tmp_path / "absent.json") is None


def test_function_terms_load_across_all_go_aspects(tmp_path: Path) -> None:
    """Terms are split by aspect in the store and must be pooled for the overlap test."""
    path = tmp_path / "function_store.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "P1": {"go_terms": {"function": ["GO:0051082"], "process": ["GO:0006457"]}},
                    "P2": {"go_terms": {"function": [], "process": []}},
                },
            },
        ),
        encoding="utf-8",
    )
    terms = load_function_terms(path)
    assert terms is not None
    assert terms["P1"] == ["GO:0006457", "GO:0051082"]
    # A protein with no terms is absent rather than present-and-empty.
    assert "P2" not in terms
