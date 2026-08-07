"""Classification metrics with interval estimates and honest baselines.

A bare accuracy number is not evidence. These helpers report per-class precision,
recall, and F1 alongside Wilson score intervals, and always evaluate the same labelled
set against trivial baselines so a reader can see how much of the performance is real
signal rather than class imbalance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# 1.959964 = the standard normal quantile for a two-sided 95% interval.
_Z_95 = 1.959964


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because the evaluation sets here are small
    and proportions sit near 0 or 1, where the naive interval misbehaves badly.
    """
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return (max(0.0, lower), min(1.0, upper))


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall, F1, and support for one class."""

    label: str
    support: int
    predicted: int
    true_positives: int

    @property
    def precision(self) -> float:
        """Fraction of predictions for this class that were correct."""
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        """Fraction of this class's members that were recovered."""
        return self.true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_json_dict(self) -> dict[str, float | int | str]:
        """Serialize for the report."""
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Full evaluation of one predictor against a labelled set."""

    name: str
    n_evaluated: int
    n_correct: int
    per_class: tuple[ClassMetrics, ...]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Overall agreement with the curated labels."""
        return self.n_correct / self.n_evaluated if self.n_evaluated else 0.0

    @property
    def accuracy_interval(self) -> tuple[float, float]:
        """95% Wilson interval for the accuracy."""
        return wilson_interval(self.n_correct, self.n_evaluated)

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1, so a rare class counts as much as a common one."""
        if not self.per_class:
            return 0.0
        return sum(metrics.f1 for metrics in self.per_class) / len(self.per_class)

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        low, high = self.accuracy_interval
        return {
            "name": self.name,
            "n_evaluated": self.n_evaluated,
            "accuracy": round(self.accuracy, 4),
            "accuracy_95ci": [round(low, 4), round(high, 4)],
            "macro_f1": round(self.macro_f1, 4),
            "per_class": [metrics.to_json_dict() for metrics in self.per_class],
            "confusion": self.confusion,
        }


def evaluate(
    name: str,
    truth: Mapping[str, str],
    predictions: Mapping[str, str],
) -> EvaluationResult:
    """Score predictions against curated labels on their shared accessions.

    Only accessions present in both mappings are evaluated, so a predictor is never
    credited or penalised for proteins it did not see.
    """
    shared = sorted(set(truth) & set(predictions))
    labels = sorted({truth[accession] for accession in shared} | {predictions[accession] for accession in shared})

    confusion: dict[str, dict[str, int]] = {
        actual: dict.fromkeys(labels, 0) for actual in sorted({truth[accession] for accession in shared})
    }
    for accession in shared:
        confusion[truth[accession]][predictions[accession]] += 1

    support = Counter(truth[accession] for accession in shared)
    predicted = Counter(predictions[accession] for accession in shared)
    correct = Counter(truth[accession] for accession in shared if truth[accession] == predictions[accession])

    per_class = tuple(
        ClassMetrics(
            label=label,
            support=support.get(label, 0),
            predicted=predicted.get(label, 0),
            true_positives=correct.get(label, 0),
        )
        for label in sorted(set(support) | set(predicted))
    )

    return EvaluationResult(
        name=name,
        n_evaluated=len(shared),
        n_correct=sum(correct.values()),
        per_class=per_class,
        confusion=confusion,
    )


def majority_baseline(truth: Mapping[str, str], accessions: Sequence[str]) -> dict[str, str]:
    """Predict the most common label for everything.

    This is the number any classifier must beat to have demonstrated anything: with an
    imbalanced set, a high accuracy can come entirely from guessing the majority class.
    """
    if not truth:
        return {}
    most_common = Counter(truth.values()).most_common(1)[0][0]
    return dict.fromkeys(accessions, most_common)
