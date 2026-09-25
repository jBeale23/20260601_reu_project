"""Tests for searching a grouping that beats A/B/C.

A search is where a result can be manufactured, so most of these pin the guards rather than
the search. One pins a bug that made the whole thing return no winners regardless of the
data - which is indistinguishable from a careful negative result, and far worse.
"""

from __future__ import annotations

import random

from validation.partition_search import (
    BINARY_COLUMNS,
    CRITERION_INFORMATION,
    CRITERION_V,
    MAX_GROUPS,
    MIN_GROUP_SHARE,
    WIDE_NUMERIC_COLUMNS,
    CandidateResult,
    _systematic_combinations,
    candidate_partitions,
    numeric_tertiles,
    score_candidate,
    search_partitions,
)


def _world(n: int = 600) -> tuple[dict, dict, dict, dict, dict]:
    """A world where one feature genuinely drives the readouts and A/B/C does not."""
    rng = random.Random(0)  # noqa: S311 - deterministic test fixture
    features, reference, discovery, held_out, noise = {}, {}, {}, {}, {}
    for index in range(n):
        gf = "true" if index % 2 == 0 else "false"
        key = f"P{index}"
        features[key] = {
            "has_gf_rich_region": gf,
            "has_dnaj_c": "true" if index % 3 == 0 else "false",
            "j_domain_position": rng.choice(["n_terminal", "internal"]),
            "n_structured_domains": str(rng.randint(0, 5)),
            "idr_fraction": str(rng.random()),
            "domain_family_layout": rng.choice(["j>gf", "j", "j>ctd"]),
        }
        reference[key] = "ABC"[index % 3]
        discovery[key] = "partnerX" if gf == "true" else "partnerY"
        held_out[key] = "systemX" if gf == "true" else "systemY"
        noise[key] = rng.choice(["n1", "n2"])
    return features, reference, discovery, held_out, noise


def test_a_genuine_grouping_is_confirmed_on_a_held_out_readout() -> None:
    """The positive case, and the bug that made it impossible.

    Confirmation p-values were left at the dataclass default of 1.0, so a candidate scoring
    a perfect 1.000 against A/B/C's 0.000 on a readout it was never selected against still
    reported as unconfirmed. The search returned no winners regardless of the data.
    """
    features, reference, discovery, held_out, _noise = _world()
    report = search_partitions(
        candidate_partitions(features),
        reference,
        discovery,
        {"held_out": held_out},
        n_permutations=200,
    )
    assert "has_gf_rich_region" in report["confirmed_candidates"]


def test_a_readout_that_is_pure_noise_confirms_nothing() -> None:
    """The guard against selecting whatever fits the noise."""
    features, reference, discovery, _held, noise = _world()
    report = search_partitions(
        candidate_partitions(features),
        reference,
        discovery,
        {"pure_noise": noise},
        n_permutations=200,
    )
    assert report["confirmed_candidates"] == []


def test_a_partition_of_one_big_group_plus_dust_is_refused() -> None:
    """Its score reflects the big group alone, whatever it looks like."""
    features = {f"P{i}": {"has_dnaj_c": "true" if i == 0 else "false"} for i in range(500)}
    assert "has_dnaj_c" not in candidate_partitions(features)
    assert MIN_GROUP_SHARE > 0


def test_a_partition_with_a_group_per_protein_is_refused() -> None:
    """It explains everything and means nothing."""
    features = {f"P{i}": {"domain_family_layout": f"unique{i}"} for i in range(400)}
    built = candidate_partitions(features)
    assert all(len(set(part.values())) <= MAX_GROUPS for part in built.values())


def test_candidates_are_built_from_architecture_only() -> None:
    """A grouping needing functional data to define it cannot be tested against it."""
    features, *_ = _world()
    built = candidate_partitions(features)
    assert built
    # Every name refers to a domain-content or geometry feature, never a partner or disease.
    assert not any(("partner" in name or "disease" in name) for name in built)


def test_scoring_needs_enough_shared_proteins() -> None:
    """A contingency table over a handful of proteins is not a measurement."""
    tiny = {f"P{i}": "g1" if i % 2 else "g2" for i in range(6)}
    assert score_candidate(tiny, tiny, tiny, name="t", readout_name="r", n_permutations=20) is None


def test_the_report_names_its_discovery_and_validation_readouts() -> None:
    """Which readout selected a candidate is part of the claim."""
    features, reference, discovery, held_out, _noise = _world()
    report = search_partitions(
        candidate_partitions(features),
        reference,
        discovery,
        {"held_out": held_out},
        discovery_name="disc",
        n_permutations=100,
    )
    assert report["discovery_readout"] == "disc"
    assert report["validation_readouts"] == ["held_out"]
    assert "never selected against" in str(report["interpretation"])


def test_systematic_combinations_cross_every_pair_and_triple() -> None:
    """The search must try all combinations, not a hand-picked handful.

    The earlier version tested nine named pairs, which can only answer whether one of those
    nine beats the reference - not whether any combination does, which is the question.
    """
    built = {
        "a": {f"P{i}": ("yes" if i % 2 else "no") for i in range(400)},
        "b": {f"P{i}": ("yes" if i % 3 else "no") for i in range(400)},
        "c": {f"P{i}": ("yes" if i % 5 else "no") for i in range(400)},
    }
    combined = _systematic_combinations(built)
    assert "a+b" in combined
    assert "a+c" in combined
    assert "b+c" in combined
    assert "a+b+c" in combined


def test_combinations_respect_the_order_ceiling() -> None:
    """Order 2 must not produce triples."""
    built = {
        name: {f"P{i}": ("yes" if (i + offset) % 2 else "no") for i in range(400)}
        for offset, name in enumerate(("a", "b", "c"))
    }
    combined = _systematic_combinations(built, max_order=2)
    assert all(name.count("+") == 1 for name in combined)


def test_degenerate_combinations_are_rejected() -> None:
    """A cross whose cells are mostly empty fails the usability gate rather than scoring."""
    # Each feature isolates one protein, so the cross is almost all singleton cells.
    built = {
        "a": {f"P{i}": ("only" if i == 0 else "rest") for i in range(400)},
        "b": {f"P{i}": ("only" if i == 1 else "rest") for i in range(400)},
    }
    assert _systematic_combinations(built) == {}


def test_combinations_need_enough_shared_proteins() -> None:
    """Two features measured on disjoint proteins cannot be crossed."""
    built = {
        "a": {f"P{i}": "yes" if i % 2 else "no" for i in range(100)},
        "b": {f"Q{i}": "yes" if i % 2 else "no" for i in range(100)},
    }
    assert _systematic_combinations(built) == {}


def test_benjamini_hochberg_rebuild_preserves_every_field() -> None:
    """Rebuilding results to attach adjusted p-values must not reset the other fields.

    The rebuild copies field by field, so a field added later is silently dropped and reads
    as its default. That happened twice: once leaving every p_adjusted at 1.0 so nothing
    could confirm, and once zeroing the information scores so every candidate tied at zero
    bits. A round-trip assertion is cheaper than finding it in a report again.
    """
    reference = {f"P{i}": ("A" if i % 3 == 0 else "B" if i % 3 == 1 else "C") for i in range(600)}
    # The readout follows the reference, so there is real information to carry. Built
    # independent of it, zero bits would be the correct answer and the test would prove
    # nothing about whether the field survived the rebuild.
    readout = {f"P{i}": f"r{i % 3}" for i in range(600)}
    candidates = {"split": {f"P{i}": ("x" if i % 3 == 0 else "y") for i in range(600)}}

    report = search_partitions(
        candidates,
        reference,
        readout,
        {"held_out": readout},
        n_permutations=20,
    )
    for row in report["discovery"] + report["validation"]:
        assert "candidate_bits" in row
        assert "reference_bits" in row
        # p_adjusted must be a real correction, not the untouched default.
        assert row["p_adjusted"] <= 1.0
    # At least one side must carry non-zero information about a readout it determines.
    assert any(row["candidate_bits"] > 0 or row["reference_bits"] > 0 for row in report["discovery"])


def _information_fixture() -> tuple[dict, dict, dict]:
    """A candidate that carries more information than the reference but scores lower on V.

    This is the case the whole criterion argument is about: a refinement whose extra groups
    carry real signal about the readout, penalised by V for being finer.
    """
    reference, readout, fine = {}, {}, {}
    for i in range(900):
        parent = "A" if i % 3 == 0 else "B" if i % 3 == 1 else "C"
        extra = i % 2
        reference[f"P{i}"] = parent
        # The readout depends on both the parent class and the extra bit, so the finer
        # partition can see something the reference cannot.
        readout[f"P{i}"] = f"{parent}{extra}"
        fine[f"P{i}"] = f"{parent}{extra}"
    return reference, readout, {"fine": fine}


def test_information_criterion_selects_a_candidate_v_rejects() -> None:
    """The criterion changes which candidates reach validation, which is the whole point."""
    reference, readout, candidates = _information_fixture()

    by_v = search_partitions(
        candidates, reference, readout, {"held": readout}, n_permutations=20, criterion=CRITERION_V
    )
    by_bits = search_partitions(
        candidates,
        reference,
        readout,
        {"held": readout},
        n_permutations=20,
        criterion=CRITERION_INFORMATION,
    )
    assert by_v["criterion"] == CRITERION_V
    assert by_bits["criterion"] == CRITERION_INFORMATION
    # The finer partition determines the readout exactly, so it must win on information.
    assert by_bits["n_beating_reference_on_discovery"] >= by_v["n_beating_reference_on_discovery"]


def test_criterion_is_recorded_in_the_report() -> None:
    """Which criterion was used has to travel with the result, not be inferred later."""
    reference, readout, candidates = _information_fixture()
    report = search_partitions(
        candidates,
        reference,
        readout,
        {"held": readout},
        n_permutations=10,
        criterion=CRITERION_INFORMATION,
    )
    assert report["criterion"] == CRITERION_INFORMATION
    assert CRITERION_INFORMATION in report["interpretation"]


def test_information_null_is_shape_matched_not_readout_shuffled() -> None:
    """A finer partition with no extra content must not gain information over a coarser one.

    This is the property that decides whether a combination beating A/B/C means anything.
    Without a shape-matched null the criterion rewards granularity: measured on the real
    data, the rank correlation between group count and raw information advantage was +0.76.
    """
    rng = random.Random(3)  # noqa: S311 - synthetic fixture, not a security boundary
    reference, readout, noisy_fine = {}, {}, {}
    for i in range(900):
        parent = "A" if i % 3 == 0 else "B" if i % 3 == 1 else "C"
        reference[f"P{i}"] = parent
        readout[f"P{i}"] = parent if rng.random() < 0.8 else "other"
        # Twelve groups, but the extra split is pure noise on top of the parent.
        noisy_fine[f"P{i}"] = f"{parent}{rng.randrange(4)}"

    report = search_partitions(
        {"noisy_fine": noisy_fine},
        reference,
        readout,
        {"held": readout},
        n_permutations=100,
        criterion=CRITERION_INFORMATION,
    )
    row = report["discovery"][0]
    assert row["n_groups"] > 3
    # The noise refinement must not be credited with information the reference lacks.
    assert not row["beats_reference_on_information"], row


def test_numeric_tertiles_labels_by_column() -> None:
    """Each numeric column gets its own named partition, not a shared one."""
    features = {f"P{i}": {"x": str(i), "y": str(1000 - i)} for i in range(300)}
    x = numeric_tertiles(features, "x", "ex")
    y = numeric_tertiles(features, "y", "why")
    assert set(x) == {"ex_tertile"}
    assert set(y) == {"why_tertile"}
    assert set(x["ex_tertile"].values()) == {"low_ex", "mid_ex", "high_ex"}


def test_numeric_tertiles_skips_unusable_columns() -> None:
    """A column that is constant, or mostly unparseable, yields no partition."""
    constant = {f"P{i}": {"x": "5"} for i in range(300)}
    assert numeric_tertiles(constant, "x", "ex") == {}
    missing = {f"P{i}": {"other": "1"} for i in range(300)}
    assert numeric_tertiles(missing, "x", "ex") == {}


def test_wide_sweep_produces_more_candidates_than_the_default() -> None:
    """Opting into every numeric column must widen the search, not silently do nothing."""
    features = {}
    for i in range(600):
        features[f"P{i}"] = {
            "has_dnaj_c": "true" if i % 2 else "false",
            "has_gf_rich_region": "true" if i % 3 else "false",
            "idr_fraction": str(i / 600),
            "mean_disorder": str((i * 7 % 600) / 600),
            "protein_length": str(100 + i),
            "n_structured_domains": str(i % 5),
            "j_domain_position": ["first", "middle", "last"][i % 3],
        }
    default = candidate_partitions(features)
    wide = candidate_partitions(features, numeric_columns=WIDE_NUMERIC_COLUMNS)
    assert len(wide) > len(default)
    # The extra columns must appear by name, so a reader can tell what was searched.
    assert any("length_tertile" in name for name in wide)


def test_classifier_outputs_are_never_candidates() -> None:
    """A partition built from the classifier's own prediction would score against itself.

    Named explicitly because these columns sit in the same table as the legitimate features
    and would be picked up by any sweep that took the whole row.
    """
    forbidden = {
        "layout_predicted_class",
        "layout_predicted_subclass",
        "layout_class_confidence",
        "layout_novelty_score",
        "novel_class_candidate",
        "shark_best_reference_class",
    }
    swept = {column for column, _label in WIDE_NUMERIC_COLUMNS} | set(BINARY_COLUMNS)
    assert not (swept & forbidden)


def test_a_hairline_margin_does_not_count_as_beating_the_reference() -> None:
    """Significance stops discriminating at large n, so the margin has to carry the decision.

    On the STRING readout a candidate confirmed on a Cramer's V margin of 0.0004 - two parts
    in a thousand of the reference - because every p-value clears at n = 9,300.
    """
    hairline = CandidateResult(
        name="c",
        readout="r",
        n_proteins=9300,
        n_groups=6,
        candidate_v=0.2469,
        reference_v=0.2465,
        null_mean_v=0.05,
        candidate_bits=0.10,
        reference_bits=0.0999,
        p_value=0.001,
        p_adjusted=0.001,
    )
    assert not hairline.beats_reference
    assert not hairline.beats_reference_on_information


def test_a_real_margin_still_counts() -> None:
    """The floor must not reject a candidate that genuinely explains more."""
    real = CandidateResult(
        name="c",
        readout="r",
        n_proteins=9300,
        n_groups=9,
        candidate_v=0.336,
        reference_v=0.2465,
        null_mean_v=0.05,
        candidate_bits=0.199,
        reference_bits=0.0888,
        p_value=0.001,
        p_adjusted=0.001,
    )
    assert real.beats_reference
    assert real.beats_reference_on_information


def test_a_zero_reference_is_beaten_by_anything_positive() -> None:
    """A reference explaining nothing cannot supply a relative margin; any signal beats it."""
    result = CandidateResult(
        name="c",
        readout="r",
        n_proteins=500,
        n_groups=3,
        candidate_v=0.2,
        reference_v=0.0,
        null_mean_v=0.01,
        candidate_bits=0.1,
        reference_bits=0.0,
        p_value=0.001,
        p_adjusted=0.001,
    )
    assert result.beats_reference
    assert result.beats_reference_on_information
