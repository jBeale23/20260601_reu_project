"""Tests for out-of-fold, genus-blocked partner prediction."""

from __future__ import annotations

import pytest

from validation.partner_prediction import (
    MIN_GROUP_TRAIN,
    blocked_folds,
    group_partner_profile,
    predict_out_of_fold,
)


def test_folds_never_split_a_block() -> None:
    """A block in two folds lets a protein be predicted by its own near-twin."""
    accessions = [f"P{i}" for i in range(60)]
    genus = {a: f"g{i % 6}" for i, a in enumerate(accessions)}
    folds = blocked_folds(accessions, genus, n_folds=3)
    seen: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for accession in fold:
            g = genus[accession]
            assert seen.setdefault(g, index) == index, f"{g} split across folds"


def test_every_accession_lands_in_exactly_one_fold() -> None:
    """Folds must partition the set, or accuracy is computed on the wrong denominator."""
    accessions = [f"P{i}" for i in range(50)]
    genus = {a: f"g{i % 7}" for i, a in enumerate(accessions)}
    folds = blocked_folds(accessions, genus, n_folds=4)
    assert sum(len(f) for f in folds) == len(accessions)
    assert set().union(*folds) == set(accessions)


def test_accessions_with_no_genus_are_still_placed() -> None:
    """A missing organism must not silently drop the protein from scoring."""
    accessions = ["P1", "P2", "P3"]
    folds = blocked_folds(accessions, {}, n_folds=2)
    assert set().union(*folds) == set(accessions)


def test_a_perfectly_predictive_partition_scores_high() -> None:
    """When group determines partner, out-of-fold prediction should recover it."""
    partition, partner, genus = {}, {}, {}
    for i in range(200):
        accession = f"P{i}"
        group = f"grp{i % 4}"
        partition[accession] = group
        partner[accession] = f"Hsp{i % 4}"
        genus[accession] = f"g{i}"  # every protein its own genus: no blocking effect
    result = predict_out_of_fold(partition, partner, genus, scheme="perfect")
    assert result.accuracy > 0.95
    assert result.lift_over_majority > 0.5


def test_a_partition_unrelated_to_partner_does_not_beat_the_majority() -> None:
    """The whole point of the lift: an uninformative carving must show near-zero."""
    partition, partner, genus = {}, {}, {}
    for i in range(400):
        accession = f"P{i}"
        partition[accession] = f"grp{i % 5}"
        # Partner depends on something the partition cannot see.
        partner[accession] = "HspA" if (i // 5) % 3 else "HspB"
        genus[accession] = f"g{i}"
    result = predict_out_of_fold(partition, partner, genus, scheme="noise")
    assert result.lift_over_majority < 0.05


def test_prediction_is_genuinely_out_of_fold() -> None:
    """A partition with one protein per group cannot predict anything.

    Fitted in sample it would score perfectly, since each group's modal partner is that one
    protein's partner. Out of fold there is nothing to learn from, so it must abstain.
    """
    partition = {f"P{i}": f"grp{i}" for i in range(60)}
    partner = {f"P{i}": f"Hsp{i}" for i in range(60)}
    genus = {f"P{i}": f"g{i}" for i in range(60)}
    result = predict_out_of_fold(partition, partner, genus, scheme="singleton")
    assert result.n_predicted == 0
    assert result.n_abstained == 60


def test_small_groups_abstain_rather_than_guess() -> None:
    """A group with two training examples is not evidence; it must decline."""
    partition, partner, genus = {}, {}, {}
    for i in range(100):
        accession = f"P{i}"
        # One large group, and a scattering of tiny ones.
        partition[accession] = "big" if i < 80 else f"tiny{i}"
        partner[accession] = "HspA"
        genus[accession] = f"g{i}"
    result = predict_out_of_fold(partition, partner, genus, scheme="mixed")
    assert result.n_abstained >= 20
    assert result.coverage < 1.0


def test_genus_blocking_lowers_accuracy_when_relatives_carry_the_signal() -> None:
    """Blocking must bite: an orthologue-driven result should fall when relatives are held out.

    Each genus has its own partner, unrelated to the partition. With random folds a protein
    is predicted by its own genus-mates; with genus blocking that shortcut is gone.
    """
    partition, partner, genus = {}, {}, {}
    for g in range(20):
        for member in range(10):
            accession = f"P{g}_{member}"
            partition[accession] = f"grp{g % 4}"
            partner[accession] = f"Hsp{g}"  # determined by genus, not by group
            genus[accession] = f"genus{g}"
    blocked = predict_out_of_fold(partition, partner, genus, scheme="blocked")
    unblocked = predict_out_of_fold(partition, partner, {a: a for a in partition}, scheme="unblocked")
    assert unblocked.accuracy > blocked.accuracy


def test_profile_reports_purity_per_group() -> None:
    """A group that is 90% one partner is a hypothesis; one split evenly is not."""
    partition, partner = {}, {}
    for i in range(20):
        partition[f"A{i}"] = "pure"
        partner[f"A{i}"] = "DnaK"
    for i in range(20):
        partition[f"B{i}"] = "mixed"
        partner[f"B{i}"] = "DnaK" if i % 2 else "HscA"
    profile = group_partner_profile(partition, partner)
    assert profile["pure"]["purity"] == pytest.approx(1.0)
    assert profile["mixed"]["purity"] == pytest.approx(0.5)


def test_profile_skips_groups_too_small_to_read() -> None:
    """A three-member group's dominant partner is not a finding."""
    partition = {f"P{i}": ("big" if i < 20 else "small") for i in range(23)}
    partner = dict.fromkeys(partition, "DnaK")
    profile = group_partner_profile(partition, partner, min_members=MIN_GROUP_TRAIN)
    assert "big" in profile
    assert "small" not in profile


def test_empty_input_yields_an_empty_result() -> None:
    """No overlap between partition and readout is zero predictions, not a crash."""
    result = predict_out_of_fold({"P1": "g"}, {"P2": "Hsp"}, {}, scheme="none")
    assert result.n_predicted == 0
    assert result.accuracy == 0.0
