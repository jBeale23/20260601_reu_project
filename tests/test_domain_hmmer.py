"""Tests for the profile-HMM homology layer.

Where pyhmmer is installed these run against the real HMMER3 engine. The point of the
layer is to be a *baseline* the layout classifier must beat, so the tests check that it
genuinely discriminates rather than that it merely returns numbers.
"""

from __future__ import annotations

import pytest

from domain_layout import hmmer
from domain_layout.hmmer import (
    BACKEND_NONE,
    BACKEND_PYHMMER,
    MIN_SEQUENCES_FOR_PROFILE,
    active_backend,
    build_profile,
    classify_by_best_profile,
    hmmer_available,
    search_profile,
)
from domain_layout.msa import align_sequences

requires_hmmer = pytest.mark.skipif(not hmmer_available(), reason="pyhmmer is not installed")

# Real J-domains: E. coli DnaJ, human DNAJA1, and yeast Ydj1 N-termini.
J_DOMAINS = (
    "MAKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG",
    "MVKETTYYDVLGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKQISQAYEVLSDAKKRELYDKGGEQ",
    "MVKETKFYDILGVPVTATDVEIKKAYRKCALKYHPDKNPSEEAAEKFKEASAAYEILSDPEKRDIYDQFG",
)
# A stretch with no J-domain character at all.
DECOY = "WWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCC"


def _built_profile() -> object:
    aligned = align_sequences(J_DOMAINS)
    return build_profile("j_domain", aligned.aligned)


def test_backend_reporting_matches_availability() -> None:
    """The reported backend must reflect what is actually installed."""
    expected = BACKEND_PYHMMER if hmmer_available() else BACKEND_NONE
    assert active_backend() == expected


def test_profile_is_refused_for_a_family_too_small_to_model() -> None:
    """Two sequences encode no positional variation; a profile would be a pairwise score."""
    assert build_profile("tiny", ["ACDEF", "ACDEF"]) is None
    assert MIN_SEQUENCES_FOR_PROFILE >= 3


def test_profile_is_refused_when_rows_are_not_an_alignment() -> None:
    """Unequal row lengths mean the caller passed unaligned sequences by mistake."""
    assert build_profile("ragged", ["ACDEF", "ACDE", "ACDEFG"]) is None


@requires_hmmer
def test_profile_length_matches_the_alignment() -> None:
    """A J-domain profile should be about the length of a J-domain, not of the alignment gaps."""
    profile = _built_profile()
    assert profile is not None
    assert profile.n_sequences == len(J_DOMAINS)
    # J-domains are roughly 65-75 residues; a wildly different M means the alignment or
    # the build consumed something it should not have.
    assert 50 <= profile.n_positions <= 90


@requires_hmmer
def test_profile_recognises_its_own_family_and_rejects_a_decoy() -> None:
    """The discrimination the baseline exists to provide."""
    profile = _built_profile()
    assert profile is not None

    hits = search_profile(profile, {"member": J_DOMAINS[0], "decoy": DECOY})
    assert "member" in hits
    assert "decoy" not in hits
    assert hits["member"].score > 0
    assert hits["member"].e_value < 1e-3


@requires_hmmer
def test_profile_detects_a_diverged_homolog_a_fixed_matrix_would_miss() -> None:
    """The reason profile HMMs are the baseline rather than pairwise similarity.

    A per-position model tolerates substitution where the family tolerates it, so a
    homolog that shares the HPD motif and little else is still found.
    """
    profile = _built_profile()
    assert profile is not None

    diverged = "MSVDPYKVLGVSRDASAAEIKKAYRQLARQYHPDVNPGDAAAEQRFKEVAEAYEVLSDPQKRAAYDRLG"
    hits = search_profile(profile, {"diverged": diverged})
    assert "diverged" in hits


@requires_hmmer
def test_search_keeps_only_the_best_hit_per_target() -> None:
    """One row per protein, or downstream joins silently multiply rows."""
    profile = _built_profile()
    assert profile is not None
    hits = search_profile(profile, dict(zip("abc", J_DOMAINS, strict=True)))
    assert len(hits) == len(set(hits))
    for name, hit in hits.items():
        assert hit.target == name
        assert hit.profile == "j_domain"


@requires_hmmer
def test_non_matching_sequences_are_absent_rather_than_zero_scored() -> None:
    """A non-match must not be confusable with a match that scored zero."""
    profile = _built_profile()
    assert profile is not None
    hits = search_profile(profile, {"decoy": DECOY})
    assert hits == {}


@requires_hmmer
def test_classification_assigns_the_class_of_the_best_scoring_profile() -> None:
    """The baseline: nearest family by bit score, with no architectural reasoning."""
    j_profile = _built_profile()
    zinc_like = (
        "CPTCHGSGAKPGTSPTTCPHCHGSGQVQMRQGFFAVQQTCPHCQGRGTLIKDPCNKCHGHGRVERSKTL",
        "CSTCNGRGAKAGSKATSCSRCGGRGVETINTGPFVMRSTCRRCGGSGEIISDPCVVCRGAGRVQKSKTL",
        "CDTCSGSGAKPGTKPKTCPTCGGSGQVTVSQGFFSVSSTCPRCHGTGKIIKEPCSVCHGKGVVEETKKV",
    )
    zinc_profile = build_profile("zinc_finger_like", align_sequences(zinc_like).aligned)
    assert j_profile is not None
    assert zinc_profile is not None

    assigned = classify_by_best_profile(
        [j_profile, zinc_profile],
        {"a_j_domain": J_DOMAINS[1], "a_zinc": zinc_like[1], "nothing": DECOY},
        {"j_domain": "J", "zinc_finger_like": "Z"},
    )
    assert assigned["a_j_domain"] == "J"
    assert assigned["a_zinc"] == "Z"
    # A sequence matching nothing is left unassigned rather than defaulted to a class,
    # which would convert the baseline's failures into successes.
    assert "nothing" not in assigned


def test_layer_degrades_when_pyhmmer_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional means optional: no import, no crash, no silent fake result."""
    monkeypatch.setattr(hmmer, "hmmer_available", lambda: False)
    assert active_backend() == BACKEND_NONE
    assert build_profile("j_domain", list(J_DOMAINS)) is None
    assert classify_by_best_profile([], {"a": J_DOMAINS[0]}, {}) == {}


def test_search_handles_empty_input() -> None:
    """Nothing to search is not an error."""
    assert search_profile.__doc__
    if hmmer_available():
        profile = _built_profile()
        assert profile is not None
        assert search_profile(profile, {}) == {}
        assert search_profile(profile, {"empty": ""}) == {}
