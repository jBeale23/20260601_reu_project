"""Tests for the disorder layer: FoldIndex fallback and the metapredict adapter."""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from domain_layout import disorder
from domain_layout.disorder import (
    BACKEND_FOLDINDEX,
    BACKEND_METAPREDICT,
    DisorderPrediction,
    complement_intervals,
    foldindex_scores,
    intervals_from_scores,
    normalized_hydropathy,
    predict_disorder,
    predict_disorder_many,
    residue_charge,
)
from tests.conftest import DNAJ_ECOLI_SEQUENCE

if TYPE_CHECKING:
    from collections.abc import Iterator

_ORDERED_SEQUENCE = "LVILVFLAVILAVLIFAVLLVIAFLVILAVFLIVALFVLIAVLFVIALVFI"
_DISORDERED_SEQUENCE = "PSEKQPSSEKQPESSKQPEEKSSQPEEKSSQPESKQEPSSKQEPSSKQEP"


class _FakeDisorderObject:
    """Minimal stand-in for metapredict's DisorderObject."""

    def __init__(self, sequence: str) -> None:
        self.sequence = sequence
        self.disorder = [0.9] * len(sequence)
        # metapredict reports Python-indexed, half-open boundaries.
        self.disordered_domain_boundaries = [[0, min(20, len(sequence))]]
        self.folded_domain_boundaries = [[min(20, len(sequence)), len(sequence)]]


@pytest.fixture
def fake_metapredict(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a fake ``metapredict`` module implementing the documented V3 API."""
    module = types.ModuleType("metapredict")
    calls: list[Any] = []

    def predict_disorder_impl(inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        calls.append((inputs, kwargs))
        if isinstance(inputs, dict):
            return {name: _FakeDisorderObject(sequence) for name, sequence in inputs.items()}
        return _FakeDisorderObject(inputs)

    module.predict_disorder = predict_disorder_impl  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metapredict", module)
    monkeypatch.setattr(disorder, "metapredict_available", lambda: True)
    return module


def test_normalized_hydropathy_bounds() -> None:
    """Hydropathy is rescaled into [0, 1] with unknown residues at the midpoint."""
    assert normalized_hydropathy("I") == pytest.approx(1.0)
    assert normalized_hydropathy("R") == pytest.approx(0.0)
    assert normalized_hydropathy("X") == pytest.approx(0.5)


def test_residue_charge_signs() -> None:
    """Only K/R are positive and D/E negative for FoldIndex."""
    assert residue_charge("K") == 1
    assert residue_charge("r") == 1
    assert residue_charge("E") == -1
    assert residue_charge("H") == 0


def test_foldindex_scores_are_bounded_and_deterministic() -> None:
    """Scores stay in [0, 1] and repeat exactly for the same input."""
    scores = foldindex_scores(DNAJ_ECOLI_SEQUENCE)
    assert len(scores) == len(DNAJ_ECOLI_SEQUENCE)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == foldindex_scores(DNAJ_ECOLI_SEQUENCE)


def test_foldindex_separates_hydrophobic_from_charged() -> None:
    """A hydrophobic stretch scores less disordered than a charged one."""
    ordered = foldindex_scores(_ORDERED_SEQUENCE)
    disordered = foldindex_scores(_DISORDERED_SEQUENCE)
    assert sum(ordered) / len(ordered) < sum(disordered) / len(disordered)


def test_foldindex_empty_sequence() -> None:
    """An empty sequence yields no scores rather than raising."""
    assert foldindex_scores("") == ()


def test_intervals_from_scores_merges_gaps_and_drops_short_runs() -> None:
    """Short gaps are closed; runs below the minimum length are discarded."""
    scores = [0.9] * 10 + [0.1] * 3 + [0.9] * 10 + [0.1] * 40 + [0.9] * 5
    intervals = intervals_from_scores(scores, threshold=0.5, min_length=12, gap_closure=10)
    assert intervals == ((1, 23),)


def test_intervals_from_scores_respects_gap_closure_limit() -> None:
    """Gaps larger than gap_closure split the runs."""
    scores = [0.9] * 15 + [0.1] * 20 + [0.9] * 15
    intervals = intervals_from_scores(scores, threshold=0.5, min_length=12, gap_closure=10)
    assert intervals == ((1, 15), (36, 50))


def test_complement_intervals_covers_remainder() -> None:
    """The complement fills exactly the residues no interval covers."""
    assert complement_intervals([(5, 10)], 20) == ((1, 4), (11, 20))
    assert complement_intervals([], 5) == ((1, 5),)
    assert complement_intervals([(1, 5)], 5) == ()
    assert complement_intervals([(1, 5)], 0) == ()


def test_prediction_helpers_report_coverage() -> None:
    """mean/fraction helpers use 1-based inclusive coordinates."""
    prediction = DisorderPrediction(
        sequence_length=10,
        scores=(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        idr_intervals=((3, 5),),
        folded_intervals=((1, 2), (6, 10)),
        backend=BACKEND_FOLDINDEX,
    )
    assert prediction.mean_over(3, 5) == pytest.approx(1.0)
    assert prediction.fraction_disordered_over(1, 10) == pytest.approx(0.3)
    assert prediction.idr_fraction == pytest.approx(0.3)
    assert prediction.residue_is_disordered() == (
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    )


def test_predict_disorder_falls_back_without_metapredict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without metapredict the FoldIndex backend is used and reported."""
    monkeypatch.setattr(disorder, "metapredict_available", lambda: False)
    prediction = predict_disorder(DNAJ_ECOLI_SEQUENCE)
    assert prediction.backend == BACKEND_FOLDINDEX
    assert prediction.sequence_length == len(DNAJ_ECOLI_SEQUENCE)
    assert disorder.active_backend() == BACKEND_FOLDINDEX


def test_predict_disorder_empty_sequence() -> None:
    """Empty input produces an empty prediction, not an error."""
    prediction = predict_disorder("")
    assert prediction.sequence_length == 0
    assert prediction.idr_intervals == ()


def test_metapredict_backend_converts_boundaries(fake_metapredict: types.ModuleType) -> None:
    """Metapredict's 0-based half-open boundaries become 1-based inclusive intervals."""
    sequence = "M" * 60
    prediction = predict_disorder(sequence)
    assert prediction.backend == BACKEND_METAPREDICT
    assert prediction.idr_intervals == ((1, 20),)
    assert prediction.folded_intervals == ((21, 60),)
    assert fake_metapredict.calls[0][1]["return_domains"] is True  # type: ignore[attr-defined]


def test_metapredict_backend_batches_sequences(fake_metapredict: types.ModuleType) -> None:
    """Many sequences are sent to metapredict in a single batched call."""
    predictions = predict_disorder_many({"a": "M" * 40, "b": "K" * 50})
    assert set(predictions) == {"a", "b"}
    assert all(prediction.backend == BACKEND_METAPREDICT for prediction in predictions.values())
    assert len(fake_metapredict.calls) == 1  # type: ignore[attr-defined]


def test_metapredict_failure_degrades_to_foldindex(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising backend is caught and the run continues with FoldIndex."""
    module = types.ModuleType("metapredict")

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "no model weights"
        raise RuntimeError(msg)

    module.predict_disorder = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metapredict", module)
    monkeypatch.setattr(disorder, "metapredict_available", lambda: True)

    prediction = predict_disorder("M" * 40)
    assert prediction.backend == BACKEND_FOLDINDEX


def test_forced_foldindex_backend_ignores_metapredict(fake_metapredict: types.ModuleType) -> None:
    """--disorder-backend foldindex never calls metapredict."""
    prediction = predict_disorder("M" * 40, backend=BACKEND_FOLDINDEX)
    assert prediction.backend == BACKEND_FOLDINDEX
    assert fake_metapredict.calls == []  # type: ignore[attr-defined]


def test_predict_disorder_many_handles_empty_values() -> None:
    """Records without sequences still get a (zero-length) prediction."""
    predictions = predict_disorder_many({"empty": "", "real": "M" * 30}, backend=BACKEND_FOLDINDEX)
    assert predictions["empty"].sequence_length == 0
    assert predictions["real"].sequence_length == 30


@pytest.mark.skipif(not disorder.metapredict_available(), reason="metapredict is not installed")
def test_real_metapredict_boundaries_tile_the_sequence() -> None:
    """Against the real package: converted intervals cover every residue exactly once.

    metapredict reports 0-based, half-open boundaries; this pins the conversion to
    1-based inclusive coordinates that the region router depends on.
    """
    sequence = DNAJ_ECOLI_SEQUENCE[:114]
    prediction = predict_disorder(sequence, backend=BACKEND_METAPREDICT)

    assert prediction.backend == BACKEND_METAPREDICT
    assert len(prediction.scores) == len(sequence)

    intervals = sorted(prediction.idr_intervals + prediction.folded_intervals)
    assert intervals[0][0] == 1
    assert intervals[-1][1] == len(sequence)
    assert sum(end - start + 1 for start, end in intervals) == len(sequence)
    for (_, previous_end), (next_start, _) in pairwise(intervals):
        assert next_start == previous_end + 1


@pytest.mark.skipif(not disorder.metapredict_available(), reason="metapredict is not installed")
def test_real_metapredict_separates_j_domain_from_gf_region() -> None:
    """The folded J-domain and the disordered G/F-rich tail are called correctly."""
    sequence = DNAJ_ECOLI_SEQUENCE[:114]
    prediction = predict_disorder(sequence, backend=BACKEND_METAPREDICT)

    # Residue 40 sits inside the J-domain; residue 100 inside the G/F-rich region.
    assert prediction.mean_over(30, 60) < prediction.mean_over(90, 114)
    assert any(start <= 100 <= end for start, end in prediction.idr_intervals)


@pytest.mark.skipif(not disorder.metapredict_available(), reason="metapredict is not installed")
def test_real_metapredict_batch_returns_every_sequence() -> None:
    """Batch prediction through the real package keeps names and lengths aligned."""
    sequences = {"short": DNAJ_ECOLI_SEQUENCE[:114], "full": DNAJ_ECOLI_SEQUENCE}
    predictions = predict_disorder_many(sequences, backend=BACKEND_METAPREDICT)

    assert set(predictions) == set(sequences)
    for name, sequence in sequences.items():
        assert predictions[name].backend == BACKEND_METAPREDICT
        assert predictions[name].sequence_length == len(sequence)


def test_disorder_prediction_is_case_insensitive() -> None:
    """Lower-case sequences predict identically; metapredict expects upper-case input."""
    sequence = DNAJ_ECOLI_SEQUENCE[:120]
    upper = predict_disorder(sequence, backend=BACKEND_FOLDINDEX)
    lower = predict_disorder(sequence.lower(), backend=BACKEND_FOLDINDEX)
    assert lower.scores == upper.scores
    assert lower.idr_intervals == upper.idr_intervals


@pytest.mark.skipif(not disorder.metapredict_available(), reason="metapredict is not installed")
def test_real_metapredict_is_case_insensitive() -> None:
    """The real backend gives the same answer for lower-case input."""
    sequence = DNAJ_ECOLI_SEQUENCE[:120]
    upper = predict_disorder(sequence, backend=BACKEND_METAPREDICT)
    lower = predict_disorder(sequence.lower(), backend=BACKEND_METAPREDICT)
    assert lower.backend == BACKEND_METAPREDICT
    assert lower.idr_intervals == upper.idr_intervals


def test_foldindex_handles_non_canonical_residues() -> None:
    """X/U/B/Z score at the hydropathy midpoint instead of raising."""
    prediction = predict_disorder("ACDEFXXXUUUBBBZZZGHIKLMNPQRSTVWY" * 3, backend=BACKEND_FOLDINDEX)
    assert prediction.sequence_length == 96
    assert all(0.0 <= score <= 1.0 for score in prediction.scores)


def test_empty_prediction_properties_are_safe() -> None:
    """Every derived statistic is defined for a zero-length prediction."""
    prediction = DisorderPrediction(
        sequence_length=0,
        scores=(),
        idr_intervals=(),
        folded_intervals=(),
        backend=BACKEND_FOLDINDEX,
    )
    assert prediction.mean_disorder == 0.0
    assert prediction.idr_fraction == 0.0
    assert prediction.mean_over(1, 10) == 0.0
    assert prediction.fraction_disordered_over(1, 10) == 0.0
    assert prediction.residue_is_disordered() == ()


def test_prediction_intervals_outside_the_sequence_are_clipped() -> None:
    """Intervals reaching past the sequence never produce out-of-range flags."""
    prediction = DisorderPrediction(
        sequence_length=5,
        scores=(1.0,) * 5,
        idr_intervals=((1, 50),),
        folded_intervals=(),
        backend=BACKEND_FOLDINDEX,
    )
    assert prediction.residue_is_disordered() == (True,) * 5


def test_intervals_from_scores_with_no_disorder() -> None:
    """A fully ordered profile yields no IDR intervals."""
    assert intervals_from_scores([0.1] * 100) == ()
    assert intervals_from_scores([]) == ()


def test_active_backend_forced_to_foldindex() -> None:
    """Explicitly selecting FoldIndex never reports metapredict."""
    assert disorder.active_backend(backend=BACKEND_FOLDINDEX) == BACKEND_FOLDINDEX


def test_metapredict_compatibility_screen() -> None:
    """Only the 20 standard residues can be one-hot encoded by metapredict."""
    assert disorder.is_metapredict_compatible("ACDEFGHIKLMNPQRSTVWY") is True
    assert disorder.is_metapredict_compatible("ACDEFX") is False  # unknown residue
    assert disorder.is_metapredict_compatible("ACDEFU") is False  # selenocysteine
    assert disorder.is_metapredict_compatible("") is False


def test_one_bad_sequence_does_not_downgrade_the_whole_batch(fake_metapredict: types.ModuleType) -> None:
    """A sequence metapredict cannot encode must not drag the batch to the fallback.

    Observed on real UniProt data: 10 sequences containing X caused 1250 of 2000
    proteins to silently fall back to FoldIndex, because predictions are batched.
    """
    sequences = {
        "clean_a": "ACDEFGHIKLMNPQRSTVWY" * 3,
        "has_x": "ACDEFGHIKXLMNPQRSTVW" * 3,
        "clean_b": "MHPDKACDEFGHIKLMNPQR" * 3,
    }
    predictions = predict_disorder_many(sequences)

    assert predictions["clean_a"].backend == BACKEND_METAPREDICT
    assert predictions["clean_b"].backend == BACKEND_METAPREDICT
    assert predictions["has_x"].backend == BACKEND_FOLDINDEX
    # The offending sequence is never handed to metapredict.
    submitted = fake_metapredict.calls[0][0]  # type: ignore[attr-defined]
    assert set(submitted) == {"clean_a", "clean_b"}


def test_batch_failure_retries_each_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a whole batch fails for another reason, sequences are retried individually."""
    module = types.ModuleType("metapredict")
    attempts: list[int] = []

    def predict_disorder_impl(inputs: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        attempts.append(len(inputs))
        if len(inputs) > 1:
            msg = "simulated batch failure"
            raise RuntimeError(msg)
        return {name: _FakeDisorderObject(sequence) for name, sequence in inputs.items()}

    module.predict_disorder = predict_disorder_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metapredict", module)
    monkeypatch.setattr(disorder, "metapredict_available", lambda: True)

    sequences = {"a": "ACDEFGHIKL" * 3, "b": "MNPQRSTVWY" * 3}
    predictions = predict_disorder_many(sequences)

    assert all(prediction.backend == BACKEND_METAPREDICT for prediction in predictions.values())
    assert attempts[0] == 2, "the batch is tried first"
    assert attempts[1:] == [1, 1], "then each sequence individually"


def test_batch_of_only_incompatible_sequences_uses_foldindex(fake_metapredict: types.ModuleType) -> None:
    """When no sequence can be encoded, metapredict is never called at all."""
    sequences = {"x1": "ACDEFGHIKX" * 4, "x2": "UUUACDEFGH" * 4}
    predictions = predict_disorder_many(sequences)

    assert all(prediction.backend == BACKEND_FOLDINDEX for prediction in predictions.values())
    assert fake_metapredict.calls == []  # type: ignore[attr-defined]
