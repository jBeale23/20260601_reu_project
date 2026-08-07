"""Tests for the alignment-free similarity layer (SHARK adapter + k-mer fallback)."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from domain_layout import shark
from domain_layout.profiles import NOVEL_SHARK_SIMILARITY
from domain_layout.shark import (
    BACKEND_BIO_SHARK,
    BACKEND_KMER,
    best_match,
    dive_like_score,
    kmer_shark_score,
    kmer_similarity,
    shark_score,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Captured before the autouse fixture can patch it, so real-backend tests can restore it.
_installed_shark_available = shark.shark_available

_GF_REGION = "GGMGGGGFGGGADFSDIFGDVFGDIFGGGRGRQRAARG"
_GF_LIKE_REGION = "GGMGGGGFGGGADFSDIFGDVFGDIFGGGRGRQRAARK"
_CHARGED_REGION = "EKEKEKDKDKEKEKDKDKEKEKDKDKEKEKDKDKEKEK"


@pytest.fixture
def fake_bio_shark(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a fake ``bio_shark.dive.run`` exposing the documented run_normal API."""
    package = types.ModuleType("bio_shark")
    dive = types.ModuleType("bio_shark.dive")
    run = types.ModuleType("bio_shark.dive.run")
    calls: list[dict[str, Any]] = []

    def run_normal(*, sequence1: str, sequence2: str, k: int, threshold: float) -> float:
        calls.append({"sequence1": sequence1, "sequence2": sequence2, "k": k, "threshold": threshold})
        return 0.42

    run.run_normal = run_normal  # type: ignore[attr-defined]
    dive.run = run  # type: ignore[attr-defined]
    package.dive = dive  # type: ignore[attr-defined]
    package.calls = calls  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "bio_shark", package)
    monkeypatch.setitem(sys.modules, "bio_shark.dive", dive)
    monkeypatch.setitem(sys.modules, "bio_shark.dive.run", run)
    monkeypatch.setattr(shark, "shark_available", lambda: True)
    return package


@pytest.fixture
def real_shark() -> None:
    """Opt out of the fallback default so a test runs against the installed bio_shark.

    Checks the captured, unpatched probe: when this fixture is pulled in dynamically the
    autouse fixture has already forced ``shark_available`` to False.
    """
    if not _installed_shark_available():
        pytest.skip("bio_shark is not installed")


@pytest.fixture(autouse=True)
def _force_kmer_backend(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Default every test to the fallback backend.

    Tests opt out by requesting ``fake_bio_shark`` (injected stub) or ``real_shark``
    (the installed package), so the default never masks a real-backend test.
    """
    opted_out = {"fake_bio_shark", "real_shark"} & set(request.fixturenames)
    if not opted_out:
        monkeypatch.setattr(shark, "shark_available", lambda: False)


def test_kmer_similarity_identity_and_range() -> None:
    """Identical k-mers score 1.0; every score stays within [0, 1]."""
    assert kmer_similarity("HPDKA", "HPDKA") == 1.0
    assert 0.0 <= kmer_similarity("HPDKA", "WWWWW") <= 1.0
    assert kmer_similarity("HPDKA", "HPD") == 0.0
    assert kmer_similarity("", "") == 0.0


def test_kmer_similarity_is_symmetric() -> None:
    """Similarity does not depend on argument order."""
    assert kmer_similarity("GGFGG", "GGMGG") == pytest.approx(kmer_similarity("GGMGG", "GGFGG"))


def test_kmer_similarity_ranks_conservative_substitutions_higher() -> None:
    """A conservative substitution scores above an unrelated k-mer."""
    conservative = kmer_similarity("KKKKK", "RRRRR")
    unrelated = kmer_similarity("KKKKK", "WWWWW")
    assert conservative > unrelated


def test_kmer_shark_score_self_similarity_is_one() -> None:
    """A region compared to itself scores 1.0."""
    assert kmer_shark_score(_GF_REGION, _GF_REGION, k=5) == pytest.approx(1.0)


def test_kmer_shark_score_is_symmetric_and_discriminative() -> None:
    """Related regions score higher than compositionally different ones."""
    forward = kmer_shark_score(_GF_REGION, _GF_LIKE_REGION, k=5)
    backward = kmer_shark_score(_GF_LIKE_REGION, _GF_REGION, k=5)
    assert forward == pytest.approx(backward)
    assert forward > kmer_shark_score(_GF_REGION, _CHARGED_REGION, k=5)


def test_kmer_shark_score_handles_short_sequences() -> None:
    """Sequences shorter than k yield no score instead of raising."""
    assert kmer_shark_score("AB", "ABCDEFG", k=5) == 0.0


def test_shark_score_clamps_k_to_shortest_sequence() -> None:
    """K is reduced to fit the shorter sequence so short IDRs still score."""
    assert shark_score("HPDKA", "HPDKA", k=20) == pytest.approx(1.0)


def test_shark_score_empty_inputs() -> None:
    """Empty regions score zero."""
    assert shark_score("", _GF_REGION) == 0.0
    assert shark_score(_GF_REGION, "") == 0.0


def test_dive_like_score_averages_k_values() -> None:
    """The dive-style aggregate is the mean over the configured k values."""
    aggregate = dive_like_score(_GF_REGION, _GF_LIKE_REGION, k_values=(3, 5))
    expected = (
        kmer_shark_score(_GF_REGION, _GF_LIKE_REGION, k=3) + kmer_shark_score(_GF_REGION, _GF_LIKE_REGION, k=5)
    ) / 2
    assert aggregate == pytest.approx(expected)


def test_best_match_picks_highest_scoring_reference() -> None:
    """best_match returns the closest reference and labels the backend used."""
    match = best_match(
        _GF_REGION,
        {"gf_like": _GF_LIKE_REGION, "charged": _CHARGED_REGION},
        k_values=(3, 5),
    )
    assert match is not None
    assert match.reference_id == "gf_like"
    assert match.backend == BACKEND_KMER
    assert 0.0 < match.score <= 1.0


def test_best_match_without_references() -> None:
    """No references means no match rather than a zero-score match."""
    assert best_match(_GF_REGION, {}) is None
    assert best_match("", {"a": _GF_REGION}) is None


def test_active_backend_reports_fallback() -> None:
    """Backend reporting reflects that bio_shark is unavailable."""
    assert shark.active_backend() == BACKEND_KMER


def test_bio_shark_backend_is_used_when_installed(fake_bio_shark: types.ModuleType) -> None:
    """With bio_shark installed, its run_normal result is returned unchanged."""
    assert shark.active_backend() == BACKEND_BIO_SHARK
    assert shark_score(_GF_REGION, _CHARGED_REGION, k=5, threshold=0.8) == pytest.approx(0.42)
    call = fake_bio_shark.calls[0]  # type: ignore[attr-defined]
    assert call["k"] == 5
    assert call["threshold"] == 0.8


def test_forced_kmer_backend_skips_bio_shark(fake_bio_shark: types.ModuleType) -> None:
    """--shark-backend blosum_kmer never calls bio_shark."""
    score = shark_score(_GF_REGION, _GF_REGION, k=5, backend=BACKEND_KMER)
    assert score == pytest.approx(1.0)
    assert fake_bio_shark.calls == []  # type: ignore[attr-defined]


def test_bio_shark_failure_degrades_to_kmer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising SHARK backend falls back to the k-mer score."""
    package = types.ModuleType("bio_shark")
    dive = types.ModuleType("bio_shark.dive")
    run = types.ModuleType("bio_shark.dive.run")

    def boom(**_kwargs: object) -> float:
        msg = "shark exploded"
        raise RuntimeError(msg)

    run.run_normal = boom  # type: ignore[attr-defined]
    dive.run = run  # type: ignore[attr-defined]
    package.dive = dive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bio_shark", package)
    monkeypatch.setitem(sys.modules, "bio_shark.dive", dive)
    monkeypatch.setitem(sys.modules, "bio_shark.dive.run", run)
    monkeypatch.setattr(shark, "shark_available", lambda: True)

    assert shark_score(_GF_REGION, _GF_REGION, k=5) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", [BACKEND_KMER, BACKEND_BIO_SHARK])
def test_scoring_is_case_insensitive(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Lower-case input must score the same as upper-case, never 0.0.

    Both backends look residues up in a substitution matrix keyed by upper-case letters.
    Before normalization a lower-case region scored 0.0 as if unrelated, which inflated
    the novelty score and could flag an ordinary JDP as a new-category candidate.
    """
    if backend == BACKEND_BIO_SHARK:
        request.getfixturevalue("real_shark")
        monkeypatch.setattr(shark, "shark_available", _installed_shark_available)
    else:
        monkeypatch.setattr(shark, "shark_available", lambda: False)

    upper = shark_score(_GF_REGION, _GF_REGION, k=5, backend=backend)
    lower = shark_score(_GF_REGION.lower(), _GF_REGION, k=5, backend=backend)
    mixed = shark_score(_GF_REGION.title(), _GF_REGION.lower(), k=5, backend=backend)

    assert upper > 0.0
    assert lower == pytest.approx(upper)
    assert mixed == pytest.approx(upper)


def test_kmer_score_is_case_insensitive() -> None:
    """The fallback normalizes case before building k-mers."""
    assert kmer_shark_score(_GF_REGION.lower(), _GF_REGION, k=5) == pytest.approx(
        kmer_shark_score(_GF_REGION, _GF_REGION, k=5),
    )


@pytest.mark.usefixtures("real_shark")
def test_real_shark_scores_are_bounded_and_discriminative() -> None:
    """Against the real package: identical > near-identical > unrelated, all within [0, 1]."""
    identical = shark_score(_GF_REGION, _GF_REGION, k=5, backend=BACKEND_BIO_SHARK)
    near = shark_score(_GF_REGION, _GF_LIKE_REGION, k=5, backend=BACKEND_BIO_SHARK)
    unrelated = shark_score(_GF_REGION, _CHARGED_REGION, k=5, backend=BACKEND_BIO_SHARK)

    assert 0.0 <= unrelated < near <= identical <= 1.0
    # The novelty term must separate related from unrelated regions on this scale.
    assert unrelated < NOVEL_SHARK_SIMILARITY < identical


@pytest.mark.usefixtures("real_shark")
def test_real_shark_survives_edge_case_inputs() -> None:
    """SHARK raises on k > sequence length and on empty input; the adapter must not."""
    # bio_shark raises "K-Mer length should not be greater than sequence length".
    assert shark_score("ACDEF", "ACDEF", k=20, backend=BACKEND_BIO_SHARK) > 0.0
    assert shark_score("", _GF_REGION, k=5, backend=BACKEND_BIO_SHARK) == 0.0
    assert shark_score(_GF_REGION, "", k=5, backend=BACKEND_BIO_SHARK) == 0.0
    assert shark_score("A", "A", k=5, backend=BACKEND_BIO_SHARK) >= 0.0


@pytest.mark.usefixtures("real_shark")
def test_real_shark_handles_non_canonical_residues() -> None:
    """Sequences with X/U/B/Z (UniProt does contain them) score without raising."""
    for sequence in ("ACDEFXXXGHIKLMNP", "ACDEFUUUGHIKLMNP", "ACDEFBZGHIKLMNPQ"):
        score = shark_score(sequence, "ACDEFGGGHIKLMNPQ", k=5, backend=BACKEND_BIO_SHARK)
        assert 0.0 <= score <= 1.0


@pytest.mark.usefixtures("real_shark")
def test_real_shark_best_match_reports_its_backend() -> None:
    """A match produced by the real package is labelled bio_shark, not the fallback."""
    match = best_match(
        _GF_REGION,
        {"gf_like": _GF_LIKE_REGION, "charged": _CHARGED_REGION},
        k_values=(3, 5),
        backend=BACKEND_BIO_SHARK,
    )
    assert match is not None
    assert match.backend == BACKEND_BIO_SHARK
    assert match.reference_id == "gf_like"


def test_kmer_similarity_with_residues_missing_from_blosum() -> None:
    """A k-mer of residues absent from BLOSUM62 scores 0.0 rather than dividing by zero."""
    assert kmer_similarity("UUUUU", "UUUUU") == 1.0
    assert kmer_similarity("UUUUU", "ACDEF") == 0.0


def test_kmer_score_with_no_shared_kmer_length() -> None:
    """Sequences shorter than k on either side score zero, not an error."""
    assert kmer_shark_score("ACD", "ACDEFGHIK", k=5) == 0.0
    assert kmer_shark_score("", "", k=5) == 0.0


def test_dive_like_score_with_empty_k_values() -> None:
    """An empty k list yields the minimum score instead of dividing by zero."""
    assert dive_like_score(_GF_REGION, _GF_LIKE_REGION, k_values=()) == 0.0


def test_best_match_is_deterministic_for_tied_scores() -> None:
    """Equal-scoring references resolve in sorted order, so runs are reproducible."""
    references = {"b_ref": _GF_REGION, "a_ref": _GF_REGION}
    first = best_match(_GF_REGION, references, k_values=(3,))
    second = best_match(_GF_REGION, references, k_values=(3,))
    assert first is not None
    assert second is not None
    assert first.reference_id == second.reference_id == "a_ref"


def test_encoding_returns_nothing_for_unusable_sequences() -> None:
    """Sequences shorter than k encode to nothing, so scoring returns the minimum."""
    assert shark._encode_kmers("ACD", 5) is None
    assert shark._encode_kmers("", 5) is None
    assert kmer_shark_score("ACD", "ACDEFGHIK", k=5) == 0.0


def test_long_region_pair_is_scored_in_blocks() -> None:
    """A pair too large for one similarity matrix is chunked without changing the score."""
    long_a = ("GGFGGSGSGGFGGSGSGGFGGSGSGGFGGSGS" * 40)[:1200]
    long_b = ("GGFGGSGSGGFGGSGSGGFGGSGAGGFGGSGS" * 40)[:1200]

    baseline = kmer_shark_score(long_a, long_b, k=5)
    original_block = shark._MAX_SIMILARITY_BLOCK
    try:
        shark._MAX_SIMILARITY_BLOCK = 500  # force multiple query blocks
        shark._encode_kmers.cache_clear()
        chunked = kmer_shark_score(long_a, long_b, k=5)
    finally:
        shark._MAX_SIMILARITY_BLOCK = original_block
        shark._encode_kmers.cache_clear()

    assert chunked == pytest.approx(baseline)


def test_non_standard_residues_are_screened_off_the_shark_backend() -> None:
    """Sequences with X/U/B/Z are not sent to bio_shark.

    bio_shark's substitution matrix holds only the twenty standard residues. Given
    anything else it writes one error line per offending k-mer pair to stdout and drops
    those pairs from the score - on a full run that produced a 191 MB log and a quietly
    degraded score. Such pairs must go to the k-mer backend, whose matrix covers them.
    """
    assert shark.is_shark_compatible("ACDEFGHIKLMNPQRSTVWY")
    for residue in ("X", "U", "B", "Z"):
        assert not shark.is_shark_compatible("MKKDY" + residue + "EILGV"), residue
    assert not shark.is_shark_compatible("")


def test_pair_backend_reflects_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported backend is the one that scored the pair, not the run-wide default."""
    monkeypatch.setattr(shark, "shark_available", lambda: True)

    clean = "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"
    unknown = "MKQDYYEILGVXKTAEEREIRKAYKRLAMKYHPDRN"

    assert shark.resolved_backend_for(clean, clean) == shark.BACKEND_BIO_SHARK
    assert shark.resolved_backend_for(unknown, clean) == shark.BACKEND_KMER
    assert shark.resolved_backend_for(clean, unknown) == shark.BACKEND_KMER
    # An explicit k-mer request stays k-mer whatever the residues are.
    assert shark.resolved_backend_for(clean, clean, backend=shark.BACKEND_KMER) == shark.BACKEND_KMER


def test_bio_shark_is_never_called_with_non_standard_residues(monkeypatch: pytest.MonkeyPatch) -> None:
    """The screen happens before the call, not inside an exception handler."""
    monkeypatch.setattr(shark, "shark_available", lambda: True)
    calls: list[tuple[str, str]] = []

    def spy(first: str, second: str, **_kwargs: object) -> float:
        calls.append((first, second))
        return 0.5

    monkeypatch.setattr(shark, "_bio_shark_score", spy)

    clean = "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"
    shark.shark_score("MKQDYYEILGVXKTAEEREIRKAYKRLAMKYHPDRN", clean)
    assert calls == []

    shark.shark_score(clean, clean)
    assert len(calls) == 1


def test_kmer_backend_scores_unknown_residues_without_error() -> None:
    """The fallback must actually handle what it is handed, not merely accept it."""
    clean = "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"
    with_unknown = clean.replace("S", "X", 1)
    score = shark.shark_score(with_unknown, clean, backend=shark.BACKEND_KMER)
    assert 0.0 < score <= 1.0
    # Identity still scores highest even with an unknown residue present.
    assert shark.shark_score(with_unknown, with_unknown, backend=shark.BACKEND_KMER) >= score


def test_best_match_reports_the_backend_that_scored_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A region with an unknown residue is labelled blosum_kmer on a bio_shark run."""
    monkeypatch.setattr(shark, "shark_available", lambda: True)
    monkeypatch.setattr(shark, "_bio_shark_score", lambda *_a, **_k: 0.9)

    references = {"ref": "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"}
    clean_match = shark.best_match("MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN", references)
    unknown_match = shark.best_match("MKQDYYEILGVXKTAEEREIRKAYKRLAMKYHPDRN", references)

    assert clean_match is not None
    assert unknown_match is not None
    assert clean_match.backend == shark.BACKEND_BIO_SHARK
    assert unknown_match.backend == shark.BACKEND_KMER
