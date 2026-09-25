"""Tests for sequence-identity redundancy control.

Genus blocking was ineffective - removing 2,174 same-genus relatives changed accuracy not
at all, because most labelled proteins sit in cross-genus ortholog groups. These tests pin
the replacement, and in particular pin that it is actually strict: a clustering that fails
to collapse near-identical sequences would reintroduce exactly the leak it exists to close.
"""

from __future__ import annotations

import random

import pytest

from validation.clustering import (
    DEFAULT_IDENTITY_THRESHOLD,
    MIN_REGION_LENGTH,
    REGION_FULL,
    REGION_J_DOMAIN,
    REGION_NON_J,
    REGIONS,
    cluster_blocked_folds,
    greedy_clusters,
    sequence_identity,
)

_JD = "MVKETKFYDILGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKEISEAYEVLSDPEKREIYD"


def test_identical_sequences_are_fully_identical() -> None:
    """The trivial case has to be exact, or every threshold is off."""
    assert sequence_identity(_JD, _JD) == pytest.approx(1.0)


def test_identity_is_normalised_by_the_shorter_sequence() -> None:
    """A domain contained in a larger protein is redundant with it, not 30% of it.

    Normalising by alignment length would call this pair distant and leave both in the
    evaluation set - the precise leak this module exists to close.
    """
    fragment = _JD[:40]
    whole = _JD + "GGGGSGGGGSGGGGS" * 6
    assert sequence_identity(fragment, whole) > 0.9


def test_unrelated_sequences_fall_below_the_threshold() -> None:
    """Poly-A against a real J-domain must not cluster."""
    assert sequence_identity("A" * 60, _JD) < DEFAULT_IDENTITY_THRESHOLD


def test_empty_sequences_are_not_identical_to_anything() -> None:
    """An empty region must never silently match."""
    assert sequence_identity("", _JD) == 0.0
    assert sequence_identity(_JD, "") == 0.0


def test_near_identical_sequences_collapse_into_one_cluster() -> None:
    """The whole point: relatives must not be split across folds."""
    sequences = {
        "P1": _JD,
        "P2": _JD[:-2] + "AA",
        "P3": "W" * len(_JD),
    }
    assignment = greedy_clusters(sequences)
    assert assignment["P1"] == assignment["P2"]
    assert assignment["P3"] != assignment["P1"]


def test_the_representative_is_the_longest_member() -> None:
    """CD-HIT's rule: keep the most informative sequence, not the first one seen."""
    long_sequence = _JD + "QQQQQQQQQQ"
    assignment = greedy_clusters({"short": _JD, "long": long_sequence})
    assert assignment["short"] == "long"
    assert assignment["long"] == "long"


def test_a_higher_threshold_never_merges_more() -> None:
    """Monotonicity - a stricter threshold cannot produce fewer clusters."""
    sequences = {f"P{index}": _JD[: 60 - index] for index in range(8)}
    loose = len(set(greedy_clusters(sequences, threshold=0.30).values()))
    strict = len(set(greedy_clusters(sequences, threshold=0.95).values()))
    assert strict >= loose


def test_folds_never_split_a_cluster() -> None:
    """If a cluster straddled a fold boundary the evaluation would leak, silently."""
    assignment = {"A1": "A1", "A2": "A1", "A3": "A1", "B1": "B1", "B2": "B1", "C1": "C1"}
    folds = cluster_blocked_folds(assignment, sorted(assignment), n_folds=3)
    for cluster in ("A1", "B1", "C1"):
        members = {key for key, rep in assignment.items() if rep == cluster}
        holding = [index for index, fold in enumerate(folds) if members & set(fold)]
        assert len(holding) == 1, f"cluster {cluster} was split across folds {holding}"


def test_every_accession_lands_in_exactly_one_fold() -> None:
    """No protein may be dropped or double-counted by the split."""
    assignment = {f"P{index}": f"R{index % 4}" for index in range(20)}
    folds = cluster_blocked_folds(assignment, sorted(assignment), n_folds=5)
    placed = [accession for fold in folds for accession in fold]
    assert sorted(placed) == sorted(assignment)
    assert len(placed) == len(set(placed))


def test_all_three_region_definitions_are_offered() -> None:
    """Full, J-domain, and non-J - the three the analysis compares."""
    assert REGIONS == (REGION_FULL, REGION_J_DOMAIN, REGION_NON_J)


def test_short_regions_are_excluded_rather_than_clustered_on_noise() -> None:
    """Below ~20 residues a few positions decide the cluster, which is not a measurement."""
    assert MIN_REGION_LENGTH >= 20


def test_random_sequences_are_not_called_redundant() -> None:
    """The regression that made the whole partition meaningless.

    Scoring matches with both gap scores at zero makes gaps free, so the aligner maximises
    matches by inserting unlimited gaps and computes a longest common subsequence rather
    than an alignment. Over a twenty-letter alphabet that puts two *random* sequences at
    0.33-0.35 identity - above the 30% threshold - so all 129 labelled proteins collapsed
    into a single cluster and cluster-blocked cross-validation became impossible.
    """
    rng = random.Random(20260822)  # noqa: S311 - deterministic test fixture
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def random_protein(length: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(length))

    for length in (60, 120, 250):
        for _ in range(4):
            identity = sequence_identity(random_protein(length), random_protein(length))
            assert identity < DEFAULT_IDENTITY_THRESHOLD, (
                f"unrelated {length}-residue sequences scored {identity:.3f}, "
                f"at or above the {DEFAULT_IDENTITY_THRESHOLD} redundancy threshold"
            )


def test_a_real_family_still_clusters_together() -> None:
    """The fix must not overshoot: genuine homologs must still be caught."""
    variant = _JD[:30] + "A" + _JD[31:]
    assert sequence_identity(_JD, variant) > 0.9


def test_clustering_a_diverse_set_yields_more_than_one_cluster() -> None:
    """One cluster over the whole set is the signature of a broken identity measure."""
    rng = random.Random(7)  # noqa: S311 - deterministic test fixture
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = {f"P{index}": "".join(rng.choice(alphabet) for _ in range(80)) for index in range(12)}
    assignment = greedy_clusters(sequences)
    assert len(set(assignment.values())) > 1


def test_a_short_region_does_not_join_a_much_longer_one() -> None:
    """Identity alone collapsed the entire labelled set into one cluster.

    Identity is normalised by the shorter sequence, which is right for asking whether a
    fragment is redundant with a protein. Under greedy clustering it also lets the single
    longest region absorb every shorter one sharing a 30%-identical stretch, however small
    a part of the long protein that stretch is. CD-HIT pairs identity with a length
    requirement for exactly this reason.
    """
    short = _JD[:40]
    long_sequence = _JD + "".join("GS" for _ in range(300))
    # Identical over the short one's whole length...
    assert sequence_identity(short, long_sequence) > 0.9
    # ...but far too short to be called redundant with it.
    assignment = greedy_clusters({"short": short, "long": long_sequence})
    assert assignment["short"] != assignment["long"]


def test_similar_lengths_still_cluster() -> None:
    """The guard must not stop genuine redundancy being caught."""
    assignment = greedy_clusters({"a": _JD, "b": _JD[:-3] + "AAA"})
    assert assignment["a"] == assignment["b"]
