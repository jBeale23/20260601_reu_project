"""Tests for the controls that decide whether an accuracy figure means anything.

A model scoring 0.93 has more than one explanation: it may be reading domain architecture,
or exploiting a shortcut that fits any labelling. These tests pin that the controls can
tell those apart - and in particular that they return a negative when they should, since a
control that always approves is decoration.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

from validation.label_controls import (
    MIN_LABELLED,
    accuracy,
    label_control_report,
    loocv,
    majority_rate,
    scrambled_label_control,
)


def _nearest_centroid(
    train: Mapping[str, Mapping[str, float]],
    train_labels: Mapping[str, str],
    test: Mapping[str, Mapping[str, float]],
) -> dict[str, str]:
    """A minimal model: assign the label of the closest class mean."""
    sums, counts = {}, {}
    for accession, values in train.items():
        label = train_labels[accession]
        counts[label] = counts.get(label, 0) + 1
        bucket = sums.setdefault(label, dict.fromkeys(values, 0.0))
        for key, value in values.items():
            bucket[key] += value
    centroids = {label: {k: v / counts[label] for k, v in bucket.items()} for label, bucket in sums.items()}
    out = {}
    for accession, values in test.items():
        best, best_distance = None, float("inf")
        for label, centre in centroids.items():
            distance = sum((values.get(k, 0.0) - v) ** 2 for k, v in centre.items())
            if distance < best_distance:
                best, best_distance = label, distance
        if best is not None:
            out[accession] = best
    return out


def _separable(n: int = 60) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Features that genuinely determine the label."""
    rng = random.Random(0)  # noqa: S311 - deterministic test fixture
    features, labels = {}, {}
    for index in range(n):
        label = "A" if index % 2 else "B"
        centre = 10.0 if label == "A" else -10.0
        features[f"P{index}"] = {"x": centre + rng.gauss(0, 0.5), "y": rng.gauss(0, 1)}
        labels[f"P{index}"] = label
    return features, labels


def _unrelated(n: int = 60) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Features carrying no information about the label."""
    rng = random.Random(1)  # noqa: S311 - deterministic test fixture
    features, labels = {}, {}
    for index in range(n):
        features[f"P{index}"] = {"x": rng.gauss(0, 1), "y": rng.gauss(0, 1)}
        labels[f"P{index}"] = "A" if index % 2 else "B"
    return features, labels


def test_a_real_signal_beats_scrambled_labels() -> None:
    """The positive case: features that determine the label must survive the control."""
    features, labels = _separable()
    result = scrambled_label_control(features, labels, _nearest_centroid, n_permutations=100)
    assert result is not None
    assert result.observed_accuracy > 0.9
    assert result.beats_scrambled
    assert result.beats_majority


def test_no_signal_fails_the_control() -> None:
    """A control that always approves is decoration.

    Features unrelated to the label must not clear the scrambled null, however the model
    happens to score.
    """
    features, labels = _unrelated()
    result = scrambled_label_control(features, labels, _nearest_centroid, n_permutations=100)
    assert result is not None
    assert not result.beats_scrambled


def test_scrambling_preserves_class_proportions() -> None:
    """The null must keep the majority-class baseline intact.

    Otherwise the comparison would confound the correspondence between features and labels
    with a change in how easy the labelling is.
    """
    features, labels = _separable()
    result = scrambled_label_control(features, labels, _nearest_centroid, n_permutations=50)
    assert result is not None
    # Proportions unchanged, so the majority rate is the same for real and scrambled.
    assert result.majority_accuracy == pytest.approx(0.5)


def test_leave_one_out_holds_out_every_protein_exactly_once() -> None:
    """No fold assignment means no variance from one, which matters at this sample size."""
    features, labels = _separable(30)
    predictions = loocv(features, labels, _nearest_centroid)
    assert set(predictions) == set(labels)


def test_a_protein_is_never_in_its_own_training_set() -> None:
    """The leak leave-one-out exists to prevent."""
    seen = {}

    def _spy(
        train: Mapping[str, Mapping[str, float]],
        train_labels: Mapping[str, str],
        test: Mapping[str, Mapping[str, float]],
    ) -> dict[str, str]:
        _ = train_labels
        for accession in test:
            seen[accession] = accession in train
        return dict.fromkeys(test, "A")

    features, labels = _separable(24)
    loocv(features, labels, _spy)
    assert not any(seen.values()), "a protein appeared in its own training set"


def test_majority_rate_is_the_floor_every_result_must_clear() -> None:
    """A set that is 70% one class makes 0.70 look like skill."""
    assert majority_rate({f"P{i}": ("A" if i < 7 else "B") for i in range(10)}) == pytest.approx(0.7)
    assert majority_rate({}) == 0.0


def test_accuracy_scores_only_predicted_proteins() -> None:
    """A model that declines a protein must not be credited or penalised for it."""
    truth = {"P1": "A", "P2": "B", "P3": "A"}
    assert accuracy(truth, {"P1": "A", "P2": "B"}) == pytest.approx(1.0)
    assert accuracy(truth, {}) == 0.0


def test_too_few_labelled_proteins_yields_no_result() -> None:
    """A permutation null over a handful of proteins is not a measurement."""
    features, labels = _separable(8)
    assert scrambled_label_control(features, labels, _nearest_centroid, n_permutations=20) is None
    assert MIN_LABELLED >= 20


def test_the_report_explains_what_scrambling_tests() -> None:
    """The reasoning is the claim; it belongs in the output."""
    features, labels = _separable()
    control = scrambled_label_control(features, labels, _nearest_centroid, n_permutations=50)
    report = label_control_report([control])
    assert report["n_beating_scrambled"] == 1
    assert "signature of a leak" in str(report["interpretation"])
