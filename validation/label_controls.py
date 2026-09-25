"""Is the classifier's accuracy real, or is something leaking?

Every supervised number in this project is scored against curated A/B/C labels, and a high
score has more than one explanation. The model might be reading domain architecture, as
claimed. It might equally be exploiting a shortcut - protein length, a feature correlated
with the label for a reason unrelated to the hypothesis, or an evaluation split that leaves
near-copies of held-out proteins in training.

Two controls separate those, and neither existed here before.

Label scrambling
----------------
Permute the labels and refit. If accuracy stays high, the model is not reading the labels'
meaning - it is exploiting structure in the *features* that happens to fit any labelling,
which is the signature of a leak. If accuracy falls to the majority-class rate, the signal
is in the correspondence between features and labels, which is what was claimed.

This is a stronger control than it first appears. It does not ask whether the model beats
chance; it asks whether the model beats *itself* trained on nonsense. A model that scores
0.93 on real labels and 0.90 on scrambled ones has learned almost nothing about the labels.

Leave-one-out
-------------
With an effective sample size of thirteen to thirty-eight after identity clustering, k-fold
splits vary a great deal depending on which proteins land together. Leave-one-out removes
that variance: every protein is held out exactly once, and the result does not depend on a
fold assignment. It is the right split at this size, and it is affordable precisely because
the labelled set is small.

An alternative to A/B/C
-----------------------
Both controls apply to any labelling, which matters because the A/B/C labels are the thing
under suspicion: a model predicting them well shows architecture recovers A/B/C, not that
A/B/C is right. Subcellular localization is an independent alternative - a cytosolic
J-domain protein does different work from an ER-lumenal one, and the label is assigned from
experiment rather than from domain content. The caveat is severe and is reported with every
result: of 59,902 proteins carrying a location, only 923 are reviewed, and localization on
an unreviewed entry is usually inferred from the same signal-peptide and transmembrane
signatures this project uses as features. Only the reviewed subset is safe.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PERMUTATIONS = 200
DEFAULT_SEED = 0
MIN_LABELLED = 20
SIGNIFICANCE_LEVEL = 0.05


class FitPredict(Protocol):
    """Train on labelled examples and predict held-out ones."""

    def __call__(
        self,
        train: Mapping[str, Mapping[str, float]],
        train_labels: Mapping[str, str],
        test: Mapping[str, Mapping[str, float]],
    ) -> dict[str, str]:
        """Return a predicted label per test accession."""
        ...


def majority_rate(labels: Mapping[str, str]) -> float:
    """Accuracy of always guessing the commonest label.

    The floor any result must clear. Reported beside every accuracy because a set that is
    70% one class makes 0.70 look like skill.
    """
    if not labels:
        return 0.0
    return Counter(labels.values()).most_common(1)[0][1] / len(labels)


def loocv(
    features: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
    fit_predict: FitPredict,
) -> dict[str, str]:
    """Out-of-fold predictions with every protein held out exactly once.

    No fold assignment means no variance from one, which matters when the effective sample
    size is in the tens.
    """
    shared = sorted(set(features) & set(labels))
    predictions: dict[str, str] = {}
    for accession in shared:
        train = {a: features[a] for a in shared if a != accession}
        train_labels = {a: labels[a] for a in shared if a != accession}
        # A fold whose training set lost the last member of a class cannot predict it; the
        # protein is skipped rather than counted as an error, which would penalise the
        # model for the split rather than for its predictions.
        if len(set(train_labels.values())) < 2:  # noqa: PLR2004 - one class predicts nothing
            continue
        predicted = fit_predict(train, train_labels, {accession: features[accession]})
        if accession in predicted:
            predictions[accession] = predicted[accession]
    return predictions


def accuracy(truth: Mapping[str, str], predictions: Mapping[str, str]) -> float:
    """Share of predicted proteins that are right."""
    shared = set(truth) & set(predictions)
    if not shared:
        return 0.0
    return sum(1 for a in shared if truth[a] == predictions[a]) / len(shared)


@dataclass(frozen=True, slots=True)
class LabelControl:
    """Observed accuracy against a scrambled-label null."""

    label_set: str
    n_labelled: int
    n_classes: int
    observed_accuracy: float
    majority_accuracy: float
    null_mean: float
    null_max: float
    null_std: float
    p_value: float
    n_permutations: int

    @property
    def z_score(self) -> float:
        """How many null standard deviations above the null mean the observation sits."""
        return (self.observed_accuracy - self.null_mean) / self.null_std if self.null_std else 0.0

    @property
    def beats_scrambled(self) -> bool:
        """Whether the model learned the labels rather than the features' own structure."""
        return self.p_value < SIGNIFICANCE_LEVEL

    @property
    def beats_majority(self) -> bool:
        """Whether it beats always guessing the commonest class."""
        return self.observed_accuracy > self.majority_accuracy

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "label_set": self.label_set,
            "n_labelled": self.n_labelled,
            "n_classes": self.n_classes,
            "observed_accuracy": round(self.observed_accuracy, 4),
            "majority_accuracy": round(self.majority_accuracy, 4),
            "scrambled_null_mean": round(self.null_mean, 4),
            "scrambled_null_max": round(self.null_max, 4),
            "z_score": round(self.z_score, 3),
            "p_value": round(self.p_value, 5),
            "beats_scrambled_labels": self.beats_scrambled,
            "beats_majority_class": self.beats_majority,
            "n_permutations": self.n_permutations,
        }


def scrambled_label_control(  # noqa: PLR0913 - features, labels, a model, and the null's name and settings
    features: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
    fit_predict: FitPredict,
    *,
    label_set: str = "labels",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> LabelControl | None:
    """Run leave-one-out on the real labels, then on scrambled ones.

    The scramble preserves the class proportions exactly - only which protein carries which
    label changes - so the null keeps the majority-class baseline intact and the comparison
    isolates the correspondence between features and labels.
    """
    shared = sorted(set(features) & set(labels))
    if len(shared) < MIN_LABELLED:
        logger.warning("only %s proteins carry both features and %s", len(shared), label_set)
        return None

    truth = {a: labels[a] for a in shared}
    observed = accuracy(truth, loocv(features, truth, fit_predict))

    rng = random.Random(seed)  # noqa: S311 - a permutation null, not a secret
    null: list[float] = []
    values = [truth[a] for a in shared]
    for _ in range(n_permutations):
        rng.shuffle(values)
        scrambled = dict(zip(shared, values, strict=True))
        null.append(accuracy(scrambled, loocv(features, scrambled, fit_predict)))

    at_least = sum(1 for value in null if value >= observed)
    return LabelControl(
        label_set=label_set,
        n_labelled=len(shared),
        n_classes=len(set(values)),
        observed_accuracy=observed,
        majority_accuracy=majority_rate(truth),
        null_mean=statistics.fmean(null) if null else 0.0,
        null_max=max(null) if null else 0.0,
        null_std=statistics.pstdev(null) if len(null) > 1 else 0.0,
        p_value=(at_least + 1) / (n_permutations + 1),
        n_permutations=n_permutations,
    )


def label_control_report(controls: Sequence[LabelControl]) -> dict[str, object]:
    """What the controls say about every labelling tested."""
    return {
        "n_label_sets": len(controls),
        "n_beating_scrambled": sum(1 for item in controls if item.beats_scrambled),
        "n_beating_majority": sum(1 for item in controls if item.beats_majority),
        "controls": [item.to_json_dict() for item in controls],
        "interpretation": (
            "Scrambling permutes labels while holding class proportions fixed, so the null "
            "keeps the majority-class baseline and the comparison isolates the "
            "correspondence between features and labels. A model scoring well on scrambled "
            "labels is exploiting structure in the features that fits any labelling - the "
            "signature of a leak - rather than reading what the labels mean. Leave-one-out "
            "is used because at an effective sample size in the tens, k-fold accuracy "
            "depends heavily on which proteins land in a fold together."
        ),
    }
