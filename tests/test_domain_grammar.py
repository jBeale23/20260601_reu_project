"""Tests for the compositional grammar layer: n-mer syntax, entropy, and complexity.

The measures here are easy to implement in a way that looks plausible and is wrong, so
each is checked against a case whose answer is known analytically or by construction
rather than against a recorded output.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from domain_layout.grammar import (
    DEFAULT_ALPHABET,
    LOW_COMPLEXITY_ORDER_SCORE,
    MIN_RELIABLE_LENGTH,
    REDUCED_ALPHABETS,
    CorpusGrammar,
    block_entropies,
    charge_profile,
    compression_ratio,
    conditional_complexity_profile,
    entropy_rates,
    nmer_counts,
    normalized_compression_distance,
    order_score,
    profile_regions,
    profile_sequence,
    reduce_sequence,
    shannon_entropy,
)

J_DOMAIN = "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG"


def _random_sequence(length: int, seed: int = 0) -> str:
    rng = random.Random(seed)  # noqa: S311 - deterministic test fixture
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))


def test_reduction_collapses_conservative_substitutions() -> None:
    """The point of a reduced alphabet is that biophysically similar residues coincide."""
    assert reduce_sequence("AVLIM") == "HHHHH"
    assert reduce_sequence("DE") == "--"
    assert reduce_sequence("KRH") == "+++"
    # A conservative substitution leaves the grammar untouched; a radical one does not.
    assert reduce_sequence("AVL") == reduce_sequence("LIM")
    assert reduce_sequence("AVL") != reduce_sequence("DEK")


def test_reduction_keeps_non_standard_residues_distinct() -> None:
    """X/U/B/Z must not silently join a real class and fake a repeat."""
    reduced = reduce_sequence("AXAUA")
    assert reduced[0] == reduced[2] == reduced[4]
    assert reduced[1] == reduced[3]
    assert reduced[1] not in {"H", "R", "P", "+", "-", "T", "C"}


def test_entropy_of_a_uniform_distribution_is_log2_of_the_support() -> None:
    """Analytic case: four equally likely symbols carry exactly two bits."""
    plugin = shannon_entropy({"a": 25, "b": 25, "c": 25, "d": 25}, corrected=False)
    assert plugin == pytest.approx(2.0)


def test_entropy_of_a_constant_sequence_is_zero() -> None:
    """One symbol carries no information."""
    assert shannon_entropy({"a": 100}, corrected=False) == pytest.approx(0.0)
    assert shannon_entropy({}) == 0.0


def test_bias_correction_raises_the_estimate_and_shrinks_with_data() -> None:
    """Plug-in entropy is biased low; the correction must go the right way and decay.

    Without this, a 30-residue linker looks more ordered than a 300-residue one purely
    because fewer observations were available to estimate it from.
    """
    small = {"a": 2, "b": 2, "c": 2, "d": 2}
    large = {"a": 200, "b": 200, "c": 200, "d": 200}

    small_gap = shannon_entropy(small) - shannon_entropy(small, corrected=False)
    large_gap = shannon_entropy(large) - shannon_entropy(large, corrected=False)

    assert small_gap > 0
    assert large_gap > 0
    assert small_gap > large_gap


def test_nmer_counts_are_overlapping_and_length_aware() -> None:
    """Overlapping windows, and nothing at all when the sequence is shorter than n."""
    assert nmer_counts("ABCD", 2) == {"AB": 1, "BC": 1, "CD": 1}
    assert sum(nmer_counts("ABCDE", 3).values()) == 3
    assert nmer_counts("AB", 5) == {}
    assert nmer_counts("ABCD", 0) == {}


def test_entropy_rates_are_non_negative_and_fall_for_a_repeat() -> None:
    """The hierarchy: a repeat grammar becomes predictable as context grows."""
    rates = entropy_rates(reduce_sequence("GGFGGM" * 30))
    assert all(rate >= 0.0 for rate in rates)
    # Once a couple of residues of context are known, a tandem repeat is determined.
    assert rates[-1] < rates[0]


def test_block_entropies_are_non_decreasing_in_order() -> None:
    """H(k) counts longer contexts, so it cannot fall as k grows."""
    blocks = block_entropies(reduce_sequence(_random_sequence(400)), max_order=4)
    for shorter, longer in pairwise(blocks):
        assert longer >= shorter - 1e-9


def test_compression_ratio_orders_repetitive_below_random() -> None:
    """The basic claim: structure compresses, noise does not."""
    repeat = reduce_sequence("GGFGGM" * 60)
    noise = reduce_sequence(_random_sequence(360))
    assert compression_ratio(repeat) < compression_ratio(noise)
    assert compression_ratio("") == 1.0


def test_order_score_is_length_invariant_for_random_sequence() -> None:
    """The reason the shuffle control exists.

    Raw compression ratio for uniformly random sequence runs from about 1.3 at 25 residues
    to about 0.5 at 1500, so thresholding it compares lengths rather than grammars. The
    shuffle-controlled score must stay near zero across that whole range.
    """
    for length in (60, 150, 400, 1200):
        score = order_score(reduce_sequence(_random_sequence(length, seed=length)))
        assert score < 0.15, f"random sequence of length {length} scored {score}"


def test_order_score_detects_a_tandem_repeat() -> None:
    """A true repeat must clear the low-complexity threshold at realistic lengths."""
    for length in (60, 150, 400):
        repeat = ("GGFGGM" * (length // 6 + 1))[:length]
        assert order_score(reduce_sequence(repeat)) >= LOW_COMPLEXITY_ORDER_SCORE


def test_order_score_ignores_composition_bias_without_order() -> None:
    """A compositionally skewed but non-periodic region is not a repeat.

    A G/F-rich linker is unusual in composition and ordinary in arrangement. Calling it
    low-complexity would flag a large fraction of every class B protein.
    """
    rng = random.Random(3)  # noqa: S311 - deterministic test fixture
    gf_like = "".join(rng.choice("GFGGFSGGAM") for _ in range(400))
    assert order_score(reduce_sequence(gf_like)) < LOW_COMPLEXITY_ORDER_SCORE


def test_order_score_is_blind_to_homopolymers_by_construction() -> None:
    """Every permutation of a homopolymer is itself, so the control cannot see it.

    This is a structural limitation, not a bug, and it is why entropy is kept as a
    separate test. Pinning it here stops anyone "fixing" the score into dishonesty.
    """
    assert order_score(reduce_sequence("Q" * 200)) == pytest.approx(0.0, abs=1e-9)
    assert profile_sequence("Q" * 200).is_compositionally_degenerate


def test_order_score_needs_enough_residues() -> None:
    """Below the reliable length the control has nothing to average over."""
    assert order_score("GGFGGM") == 0.0
    assert order_score(reduce_sequence("GGFGGM" * 2)) == 0.0


def test_order_score_is_reproducible_from_the_seed() -> None:
    """Shuffling is random, so the score must still be deterministic."""
    sequence = reduce_sequence("GGFGGMASD" * 30)
    assert order_score(sequence, seed=11) == order_score(sequence, seed=11)


def test_the_two_complexity_tests_cover_different_failures() -> None:
    """Neither measure alone catches both degenerate cases."""
    homopolymer = profile_sequence("Q" * 120)
    repeat = profile_sequence("GGFGGM" * 25)
    ordinary = profile_sequence(J_DOMAIN)

    # Entropy catches the homopolymer that the shuffle control cannot see.
    assert homopolymer.is_compositionally_degenerate
    assert not homopolymer.is_low_complexity
    # The shuffle control catches the repeat.
    assert repeat.is_low_complexity
    # An ordinary folded domain trips neither.
    assert not ordinary.is_low_complexity
    assert not ordinary.is_compositionally_degenerate


def test_normalized_compression_distance_is_a_sensible_metric() -> None:
    """Near zero to itself, much larger between unrelated sequences, bounded in [0, 1].

    "Near" rather than "equal": NCD reaches exactly zero only for an ideal compressor.
    A real one pays a small constant overhead, which is relatively largest on the most
    compressible inputs - so the self-distance of a tandem repeat is around 0.12, not 0.
    What must hold is the ordering, and that it holds by a wide margin.
    """
    repeat = "GGFGGM" * 40
    other = _random_sequence(240, seed=7)

    self_distance = normalized_compression_distance(repeat, repeat)
    cross_distance = normalized_compression_distance(repeat, other)
    assert self_distance < 0.25
    assert cross_distance > 3 * self_distance
    for first, second in ((repeat, other), (other, repeat), ("", ""), ("A" * 30, "")):
        assert 0.0 <= normalized_compression_distance(first, second) <= 1.0
    assert normalized_compression_distance("", "") == 0.0
    assert normalized_compression_distance("ACDE", "") == 1.0


def test_normalized_entropy_is_bounded_and_scaled_to_the_alphabet() -> None:
    """Comparable across alphabets in a way raw bits are not."""
    for alphabet in REDUCED_ALPHABETS:
        profile = profile_sequence(_random_sequence(400), alphabet=alphabet)
        assert 0.0 <= profile.normalized_entropy <= 1.0
    assert profile_sequence("Q" * 200).normalized_entropy == pytest.approx(0.0, abs=1e-6)


def test_profile_reports_reliability_rather_than_hiding_it() -> None:
    """A measurement from too little sequence is reported, not silently returned."""
    short = profile_sequence("MKQDYYEIL")
    assert not short.is_reliable
    # A short region can never be called low-complexity on this evidence.
    assert not short.is_low_complexity
    assert profile_sequence(J_DOMAIN).is_reliable
    assert MIN_RELIABLE_LENGTH > 0


def test_profile_serializes_every_reported_measure() -> None:
    """Anything the report claims must actually be in the payload."""
    payload = profile_sequence(J_DOMAIN).to_json_dict()
    for key in (
        "length",
        "alphabet",
        "block_entropies",
        "entropy_rates",
        "normalized_entropy",
        "compression_ratio",
        "order_score",
        "dominant_nmer",
        "is_low_complexity",
        "is_compositionally_degenerate",
        "is_reliable",
    ):
        assert key in payload, key
    assert payload["alphabet"] == DEFAULT_ALPHABET


def test_profile_regions_skips_empty_sequences() -> None:
    """Region tables carry empty rows; they are not grammars."""
    profiles = profile_regions([("a", J_DOMAIN), ("b", "")])
    assert set(profiles) == {"a"}


def test_corpus_grammar_scores_an_outlier_above_the_background() -> None:
    """The anomaly detector must rank an alien grammar above ordinary members."""
    rng = random.Random(5)  # noqa: S311 - deterministic test fixture
    corpus = ["".join(rng.choice("GFGGFSGGAM") for _ in range(200)) for _ in range(60)]
    model = CorpusGrammar(order=3).fit(corpus)

    typical = model.score(corpus[0])
    alien = model.score("KRKRKRDEDEDEKRKRKRDEDEDE" * 8)

    assert alien.surprisal_per_residue > typical.surprisal_per_residue
    assert alien.percentile > typical.percentile
    assert alien.z_score > typical.z_score


def test_corpus_grammar_never_returns_infinite_surprisal() -> None:
    """An unseen n-mer must not send an otherwise ordinary region to infinity.

    This is what the Laplace smoothing is for; without it a single novel n-mer dominates
    the whole score.
    """
    model = CorpusGrammar(order=3).fit(["ACDEFGHIKLMNPQRSTVWY" * 10])
    score = model.score("WWWWWWWWWWCCCCCCCCCC" * 5)
    assert math.isfinite(score.surprisal_per_residue)
    assert score.surprisal_per_residue > 0


def test_unfitted_corpus_grammar_scores_nothing() -> None:
    """A model with no background cannot pretend to detect an anomaly."""
    model = CorpusGrammar()
    assert not model.is_fitted
    score = model.score(J_DOMAIN)
    assert score.surprisal_per_residue == 0.0
    assert score.percentile == 0.0
    # Fitting on nothing usable leaves it unfitted rather than silently ready.
    assert not CorpusGrammar().fit(["", "AB"]).is_fitted


def test_corpus_grammar_is_reproducible() -> None:
    """Same corpus, same scores - there is no hidden state between fits."""
    corpus = [_random_sequence(150, seed=index) for index in range(20)]
    first = CorpusGrammar(order=2).fit(corpus).score(J_DOMAIN)
    second = CorpusGrammar(order=2).fit(corpus).score(J_DOMAIN)
    assert first.surprisal_per_residue == pytest.approx(second.surprisal_per_residue)


def test_corpus_grammar_percentile_is_bounded() -> None:
    """A percentile outside [0, 1] would be meaningless in the report."""
    corpus = [_random_sequence(200, seed=index) for index in range(30)]
    model = CorpusGrammar().fit(corpus)
    for sequence in (J_DOMAIN, "Q" * 200, corpus[0]):
        assert 0.0 <= model.score(sequence).percentile <= 1.0


def test_repeating_earlier_sequence_costs_almost_nothing() -> None:
    """The defining property of a *conditional* measure.

    A stretch identical to something already seen carries no new information, so its
    windows must cost near zero bits. An unconditional measure would score the second copy
    exactly as it scored the first.
    """
    profile = conditional_complexity_profile(J_DOMAIN + J_DOMAIN)
    assert profile.trough == pytest.approx(0.0, abs=1e-9)
    # The first copy is not free; only the repeat is.
    assert profile.peak > 1.0


def test_novel_sequence_stays_expensive_throughout() -> None:
    """Nothing repeats, so no window should collapse to zero."""
    profile = conditional_complexity_profile(_random_sequence(400, seed=11))
    assert profile.trough > 0.0


def test_profile_is_normalised_against_absolute_length() -> None:
    """The fix for the leak: a longer protein must not score higher for being longer.

    The challenger's features previously included raw length, and class C is largely
    "J-domain only" - i.e. shorter - so the model could learn length instead of grammar.
    """
    short = conditional_complexity_profile(_random_sequence(120, seed=3))
    long_one = conditional_complexity_profile(_random_sequence(1200, seed=3))
    # Bits per residue, not bits per protein, so the two are comparable.
    assert abs(short.mean - long_one.mean) < 2.0
    assert short.dynamic_range >= 0.0
    assert long_one.dynamic_range >= 0.0


def test_dynamic_range_is_stable_on_repetitive_sequence() -> None:
    """The replacement for a peak-over-mean ratio, which was unusable.

    Dividing by the sequence's own mean explodes when the mean approaches zero, which is
    what happens on a highly repetitive protein - and repetitive is exactly what class B's
    G/F-rich regions are, so the instability would have struck where it mattered most.
    A difference of two per-residue quantities cannot blow up that way.
    """
    mild = conditional_complexity_profile(J_DOMAIN * 2)
    extreme = conditional_complexity_profile(J_DOMAIN * 20)
    # Ten times more repetition must not move the statistic by an order of magnitude.
    assert abs(extreme.dynamic_range - mild.dynamic_range) < 2.0
    assert extreme.dynamic_range >= 0.0


def test_spread_separates_uniform_from_mixed_grammar() -> None:
    """A protein that switches grammar mid-sequence has a wide profile."""
    uniform = conditional_complexity_profile(_random_sequence(300, seed=7))
    mixed = conditional_complexity_profile(_random_sequence(150, seed=7) + "GGFGGM" * 30)
    assert mixed.spread > uniform.spread


def test_windows_overlap_so_the_profile_is_smoothed_by_construction() -> None:
    """Overlap rather than a filter.

    The noise here is autocorrelated and bounded below by the compressor's block
    structure, so Savitzky-Golay or Gaussian smoothing would impose an assumption the data
    does not satisfy. Overlapping windows average without assuming anything.
    """
    profile = conditional_complexity_profile(J_DOMAIN * 4, window=20, step=5)
    assert profile.step < profile.window
    gaps = [second - first for first, second in pairwise(profile.starts)]
    assert set(gaps) == {profile.step}


def test_profile_is_empty_for_a_sequence_shorter_than_two_windows() -> None:
    """Nothing to profile is not an error, and must not fabricate a value.

    Two windows, not one: the first is spent establishing the prefix the rest are
    conditioned on.
    """
    profile = conditional_complexity_profile("MKQD" * 6, window=20)
    assert profile.bits_per_residue == ()
    assert profile.mean == 0.0
    assert profile.peak == 0.0
    assert profile.spread == 0.0
    assert profile.dynamic_range == 0.0


def test_profile_never_returns_negative_bits() -> None:
    """A compressor can shrink on more input at a block boundary; that is an artefact."""
    for sequence in (J_DOMAIN * 6, "Q" * 300, _random_sequence(500, seed=13)):
        assert all(value >= 0.0 for value in conditional_complexity_profile(sequence).bits_per_residue)


def test_complexity_profile_serializes_every_reported_measure() -> None:
    """Anything claimed in the report must be in the payload."""
    payload = conditional_complexity_profile(J_DOMAIN * 3).to_json_dict()
    for key in (
        "window",
        "step",
        "length",
        "mean_bits_per_residue",
        "peak_bits_per_residue",
        "spread",
        "dynamic_range",
    ):
        assert key in payload, key


def test_net_charge_alone_cannot_distinguish_these_three() -> None:
    """The reason local charge is measured alongside global.

    A block of lysines followed by a block of aspartates, the same residues alternating,
    and a polarised sequence with a neutral spacer all have net charge zero. Only the
    windowed measure separates them, and only the segregated ones behave as polyampholytes.
    """
    blocky = charge_profile("K" * 30 + "D" * 30)
    mixed = charge_profile("KD" * 30)
    polarised = charge_profile("R" * 20 + "A" * 20 + "E" * 20)

    assert blocky.net_charge == mixed.net_charge == polarised.net_charge == 0.0
    assert mixed.charge_segregation == pytest.approx(0.0, abs=1e-9)
    assert blocky.charge_segregation > 0.5
    assert polarised.charge_segregation > 0.5


def test_segregation_normalises_for_how_many_residues_are_charged() -> None:
    """Few scattered charges must not read as a segregated sequence.

    Dividing the window spread by the charged fraction separates "few charges spread
    thinly" from "many charges gathered into opposing blocks"; a raw spread confuses them.
    """
    sparse = charge_profile("A" * 28 + "K" + "A" * 28 + "D")
    dense_blocks = charge_profile("K" * 30 + "D" * 30)
    assert dense_blocks.charge_segregation > sparse.charge_segregation


def test_charge_is_measured_at_physiological_ph() -> None:
    """His is left neutral; its pKa near 6 makes it only partly protonated at pH 7.4."""
    assert charge_profile("H" * 20).net_charge == 0.0
    assert charge_profile("K" * 20).net_charge == 20.0
    assert charge_profile("D" * 20).net_charge == -20.0


def test_charge_measures_are_length_normalised() -> None:
    """Everything but raw net charge must be comparable across protein sizes."""
    short = charge_profile("KRDE" * 10)
    long_one = charge_profile("KRDE" * 100)
    assert short.net_charge_per_residue == pytest.approx(long_one.net_charge_per_residue, abs=1e-9)
    assert short.fraction_charged == pytest.approx(long_one.fraction_charged, abs=1e-9)
    assert short.charge_segregation == pytest.approx(long_one.charge_segregation, abs=0.1)


def test_charge_profile_handles_degenerate_input() -> None:
    """Empty and uncharged sequences are not errors."""
    empty = charge_profile("")
    assert empty.length == 0
    assert empty.charge_segregation == 0.0
    uncharged = charge_profile("A" * 50)
    assert uncharged.fraction_charged == 0.0
    assert uncharged.charge_segregation == 0.0


def test_window_wider_than_the_sequence_is_clamped() -> None:
    """A short region must still yield a profile rather than an empty one."""
    profile = charge_profile("KRDE", window=100)
    assert profile.window == 4
    assert profile.net_charge == 0.0


def test_charge_profile_serializes_every_reported_measure() -> None:
    """Anything the report claims must be in the payload."""
    payload = charge_profile(J_DOMAIN).to_json_dict()
    for key in (
        "net_charge",
        "net_charge_per_residue",
        "fraction_charged",
        "mean_window_charge",
        "max_window_charge",
        "min_window_charge",
        "charge_segregation",
    ):
        assert key in payload, key
