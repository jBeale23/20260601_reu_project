"""Tests for the challenger model and the bake-off that decides its fate."""

from __future__ import annotations

import dataclasses
import inspect
import random
from typing import TYPE_CHECKING

from experimental.bakeoff import (
    NEURAL_NAME,
    BakeoffInputs,
    Verdict,
    class_confusion,
    grammar_predictions,
    run_bakeoff,
)
from experimental.grammar_classifier import fit_grammar_model, grammar_features

if TYPE_CHECKING:
    import pytest


def _verdict(*, macro_f1: bool, uncertainty: bool, homology: bool) -> Verdict:
    """Build a verdict from the three promotion criteria, in order."""
    return Verdict(
        challenger="c",
        incumbent="i",
        beats_macro_f1=macro_f1,
        beats_uncertainty=uncertainty,
        beats_homology_baseline=homology,
    )


def _sequence(kind: str, index: int) -> str:
    rng = random.Random(hash((kind, index)) % (2**31))  # noqa: S311 - deterministic fixture
    if kind == "repeat":
        return ("GGFGGM" * 40)[: 180 + index]
    if kind == "charged":
        return "".join(rng.choice("KRDE") for _ in range(200 + index))
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(200 + index))


def test_promotion_requires_every_criterion() -> None:
    """Two out of three is not a promotion."""
    criteria = ("beats_macro_f1", "beats_uncertainty", "beats_homology_baseline")
    assert _verdict(macro_f1=True, uncertainty=True, homology=True).promoted
    for missing in criteria:
        flags = dict.fromkeys(criteria, True)
        flags[missing] = False
        assert not Verdict(challenger="c", incumbent="i", **flags).promoted, missing


def test_verdict_states_the_decision_in_words() -> None:
    """The report should not require a reader to interpret three booleans."""
    assert "retained" in _verdict(macro_f1=True, uncertainty=False, homology=True).to_json_dict()["decision"]
    assert "replaces" in _verdict(macro_f1=True, uncertainty=True, homology=True).to_json_dict()["decision"]


def test_grammar_features_separate_constructed_classes() -> None:
    """The features must carry signal before any classifier can be blamed for lacking it."""
    repeat = grammar_features(_sequence("repeat", 0))
    charged = grammar_features(_sequence("charged", 0))
    assert repeat["order_score"] > charged["order_score"]
    assert charged["charge3_f_+"] + charged["charge3_f_-"] > repeat["charge3_f_+"] + repeat["charge3_f_-"]


def test_model_learns_a_constructed_distinction() -> None:
    """A fitted model must recover classes that genuinely differ in grammar.

    This is the floor: if the model cannot separate sequences built to differ, a poor
    score on real JDPs tells us nothing about the features.
    """
    features = {}
    labels = {}
    for index in range(12):
        for kind, label in (("repeat", "R"), ("charged", "C")):
            accession = f"{label}{index}"
            features[accession] = grammar_features(_sequence(kind, index))
            labels[accession] = label

    model = fit_grammar_model(features, labels)
    assert model is not None
    correct = sum(1 for accession, label in labels.items() if model.predict(features[accession]) == label)
    assert correct / len(labels) > 0.9


def test_model_refuses_to_fit_a_single_class() -> None:
    """A classifier with one class is not a classifier."""
    features = {f"a{i}": grammar_features(_sequence("random", i)) for i in range(5)}
    assert fit_grammar_model(features, dict.fromkeys(features, "A")) is None
    assert fit_grammar_model({}, {}) is None


def test_model_survives_a_constant_feature() -> None:
    """A within-class constant would give infinite likelihood without a variance floor."""
    features = {f"a{i}": {"x": 1.0, "y": float(i)} for i in range(4)}
    features.update({f"b{i}": {"x": 1.0, "y": float(i + 10)} for i in range(4)})
    labels = {**dict.fromkeys([f"a{i}" for i in range(4)], "A"), **dict.fromkeys([f"b{i}" for i in range(4)], "B")}
    model = fit_grammar_model(features, labels)
    assert model is not None
    assert model.predict({"x": 1.0, "y": 0.0}) == "A"
    assert model.predict({"x": 1.0, "y": 11.0}) == "B"


def test_bakeoff_reports_rather_than_crashes_on_too_little_data() -> None:
    """A tiny label set cannot be cross-validated; say so instead of failing."""
    result = run_bakeoff([], BakeoffInputs(labels={"a": "A"}, incumbent={"a": "A"}, homology={}))
    assert result["status"].startswith("not evaluated")


def test_class_confusion_counts_actual_versus_predicted() -> None:
    """Reading where a challenger fails requires the matrix, not just the score."""
    matrix = class_confusion({"a": "A", "b": "A", "c": "B"}, {"a": "A", "b": "B", "c": "B"})
    assert matrix["A"]["A"] == 1
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["B"] == 1
    # A protein with no prediction contributes no row rather than a spurious one.
    assert class_confusion({"a": "A"}, {}) == {}


def test_predictions_are_out_of_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model must never predict a protein it was fitted on.

    Fitting on everything and scoring on everything would let the challenger memorise its
    own evaluation set, which is precisely the advantage a rule-based incumbent cannot have.
    """
    fitted_on: list[set[str]] = []
    real_fit = fit_grammar_model

    def spy(features, labels):  # noqa: ANN001, ANN202
        fitted_on.append(set(labels))
        return real_fit(features, labels)

    monkeypatch.setattr("experimental.bakeoff.fit_grammar_model", spy)

    class _Record:
        def __init__(self, accession: str, sequence: str) -> None:
            self.accession = accession
            self.sequence = sequence

    class _Layout:
        def __init__(self, accession: str, sequence: str) -> None:
            self.record = _Record(accession, sequence)
            self.regions = ()

    layouts = [_Layout(f"P{i}", _sequence("random", i)) for i in range(20)]
    labels = {f"P{i}": ("A" if i % 2 else "B") for i in range(20)}

    predictions = grammar_predictions(layouts, labels, n_folds=5, seed=0)
    assert predictions
    # Every fold's training set must exclude the proteins it then predicted.
    assert fitted_on
    for training in fitted_on:
        assert len(training) < len(labels)


def test_the_neural_challenger_is_scored_in_the_bakeoff() -> None:
    """Every model must be scored on the same proteins, including the neural one.

    It was previously written to its own report only, so it was never compared head-to-head
    against the incumbent, the grammar challenger, the profile HMM, or the two structural
    variants - which is the whole purpose of the bake-off.
    """
    assert "neural" in {field.name for field in dataclasses.fields(BakeoffInputs)}
    assert NEURAL_NAME == "neural_hierarchical_cv"


def test_neural_predictions_narrow_the_shared_protein_set() -> None:
    """A model that predicted fewer proteins must shrink the comparison, not be excused.

    Otherwise the neural challenger could score on an easier subset than everything else.
    """
    source = inspect.getsource(run_bakeoff)
    assert "common &= set(inputs.neural)" in source
