"""Would the ranking survive plausible label errors? A check that assumes nothing.

:mod:`validation.capacity` reports a proved bound on how many methods a noisy benchmark can
order. That bound is machine-checked, but it is also *external* to this project, and a
conclusion resting solely on a proof nobody here has verified is a conclusion resting on
trust.

This module answers the same question empirically and independently. Flip a plausible
fraction of the labels at random, re-score every method, and count how often the ordering
between two of them reverses. No theorem is assumed; the only input is a guess at the error
rate, and that guess is swept rather than fixed.

They do not agree, and the reason is placement, not slack in the proof
---------------------------------------------------------------------
Measured on this project's own numbers - a 0.0387 accuracy gap over 129 labels - the bound
says the ordering needs label error below 1.93%, while the simulation finds it survives to
8%. Both are right, and the difference is not slack that a sharper reading recovers.

The obvious suspect was that the bound is applied in its crude form, ``c = 2 * eps``, while
the source's impossibility construction (``unresolvable_pair``) is conditioned on the
tighter ``2 * min(nu, |errors_worse minus errors_better|)`` - twice the count of targets the
losing method fails *and the winner does not*. Targets both methods already get wrong shift
both scores together and cannot reverse anything, so charging the margin against all of
``nu`` looked wasteful.

It is not. That sharper form is implemented in :func:`validation.capacity.sharp_decidability`
and is provably equivalent, because the margin can never exceed the exploitable count:
``margin = exploitable - |errors_better minus errors_worse|``. If the budget covers the
exploitable set both forms refuse to certify; if it does not, the sharp condition collapses
to ``margin >= 2 * budget``, which is the crude bound exactly. Twenty thousand random pairs
find no case where they disagree.

So the whole discrepancy is *placement*. The bound lets an adversary put every one of its
``nu`` errors on the targets where they reverse the pair; the simulation draws them at
random, and most land where they do nothing. That is the difference between asking whether
an ordering is **certifiable** against any admissible relabelling and whether it is
**stable** under the errors one actually expects.

Reporting either alone would mislead, in opposite directions. The bound alone reads as "this
ranking is worthless" when in practice it is fairly robust; the simulation alone reads as
"this ranking is settled" when it cannot in fact be certified. Both are reported, and the
word used for each - certifiable versus stable - is chosen to keep them apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Error rates swept. The lowest is optimistic for any curated resource; 0.08 is what a
# comparable curated benchmark measures for itself.
DEFAULT_ERROR_RATES = (0.01, 0.02, 0.05, 0.08, 0.10)

# Resamples per rate. A thousand resolves a flip probability to about one percent, which is
# finer than the decision needs.
DEFAULT_RESAMPLES = 1000

# Flip probability above which an ordering is called unreliable. One reversal in twenty
# resamples is the same tolerance the significance threshold uses elsewhere.
UNSTABLE_FLIP_RATE = 0.05

# The error rate the headline summary is taken at, matching what a comparable curated
# benchmark measures for itself well enough to be a fair default.
HEADLINE_ERROR_RATE = 0.05


@dataclass(frozen=True, slots=True)
class OrderingStability:
    """How often one pairwise ordering survives resampled label error."""

    better: str
    worse: str
    error_rate: float
    observed_gap: float
    flip_rate: float
    n_resamples: int

    @property
    def is_stable(self) -> bool:
        """Whether the ordering holds under this much label error."""
        return self.flip_rate < UNSTABLE_FLIP_RATE

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "better": self.better,
            "worse": self.worse,
            "assumed_error_rate": self.error_rate,
            "observed_gap": round(self.observed_gap, 4),
            "flip_rate": round(self.flip_rate, 4),
            "n_resamples": self.n_resamples,
            "stable": self.is_stable,
        }


def _accuracy(truth: Mapping[str, str], predictions: Mapping[str, str], keys: Sequence[str]) -> float:
    scored = [key for key in keys if key in predictions]
    if not scored:
        return 0.0
    return sum(1 for key in scored if predictions[key] == truth[key]) / len(scored)


def _corrupt(
    truth: Mapping[str, str],
    labels: Sequence[str],
    error_rate: float,
    rng: random.Random,
) -> dict[str, str]:
    """Flip each label to a different class with probability ``error_rate``.

    Flipping to a *different* class rather than a random one models annotation error as it
    actually occurs: a curator assigns the wrong class, not a nonexistent one.
    """
    corrupted: dict[str, str] = {}
    for accession, actual in truth.items():
        if rng.random() < error_rate:
            alternatives = [label for label in labels if label != actual]
            corrupted[accession] = rng.choice(alternatives) if alternatives else actual
        else:
            corrupted[accession] = actual
    return corrupted


@dataclass(frozen=True, slots=True)
class ResampleSettings:
    """How the label perturbation is swept, and the seed that fixes it."""

    error_rates: Sequence[float] = DEFAULT_ERROR_RATES
    n_resamples: int = DEFAULT_RESAMPLES
    seed: int = 0


DEFAULT_RESAMPLE_SETTINGS = ResampleSettings()


def ordering_stability(
    truth: Mapping[str, str],
    first: Mapping[str, str],
    second: Mapping[str, str],
    *,
    names: tuple[str, str],
    settings: ResampleSettings = DEFAULT_RESAMPLE_SETTINGS,
) -> list[OrderingStability]:
    """How often two methods swap places when labels are perturbed.

    Both are scored on the same proteins throughout, so a reversal can only come from the
    labels moving, never from one method being asked an easier question.
    """
    first_name, second_name = names
    shared = sorted(set(truth) & set(first) & set(second))
    if not shared:
        return []

    labels = sorted(set(truth.values()))
    base_first = _accuracy(truth, first, shared)
    base_second = _accuracy(truth, second, shared)
    leads_first = base_first >= base_second
    better, worse = (first_name, second_name) if leads_first else (second_name, first_name)
    leader, trailer = (first, second) if leads_first else (second, first)
    gap = abs(base_first - base_second)

    results: list[OrderingStability] = []
    for error_rate in settings.error_rates:
        # Deterministic, seeded resampling for reproducibility; not a security context.
        rng = random.Random(settings.seed)  # noqa: S311
        flips = 0
        for _ in range(settings.n_resamples):
            corrupted = _corrupt(truth, labels, error_rate, rng)
            if _accuracy(corrupted, trailer, shared) > _accuracy(corrupted, leader, shared):
                flips += 1
        results.append(
            OrderingStability(
                better=better,
                worse=worse,
                error_rate=error_rate,
                observed_gap=gap,
                flip_rate=flips / settings.n_resamples,
                n_resamples=settings.n_resamples,
            ),
        )
    return results


def stability_report(
    truth: Mapping[str, str],
    predictions: Mapping[str, Mapping[str, str]],
    *,
    settings: ResampleSettings = DEFAULT_RESAMPLE_SETTINGS,
) -> dict[str, object]:
    """Pairwise ordering stability for every pair of methods.

    Reported beside the proved capacity bound. The two are independent routes to the same
    question, so agreement strengthens the claim and disagreement is itself worth knowing.
    """
    method_names = sorted(predictions)
    pairs: list[dict[str, object]] = []
    for index, first_name in enumerate(method_names):
        for second_name in method_names[index + 1 :]:
            pairs.extend(
                item.to_json_dict()
                for item in ordering_stability(
                    truth,
                    predictions[first_name],
                    predictions[second_name],
                    names=(first_name, second_name),
                    settings=settings,
                )
            )

    unstable = [item for item in pairs if item["assumed_error_rate"] == HEADLINE_ERROR_RATE and not item["stable"]]
    return {
        "n_methods": len(method_names),
        "n_pairwise_tests": len(pairs),
        "error_rates_swept": list(settings.error_rates),
        "n_resamples": settings.n_resamples,
        "n_unstable_at_headline_error": len(unstable),
        "headline_error_rate": HEADLINE_ERROR_RATE,
        "pairs": pairs,
        "interpretation": (
            "Labels are flipped at the assumed rate and every method re-scored; the flip "
            "rate is how often the ordering reverses. This assumes no theorem - it is a "
            "simulation - and is reported beside the proved capacity bound as an "
            "independent route to the same question, and the two are expected to differ. "
            "The bound is worst case - whether any arrangement of errors within budget "
            "could reverse the ordering, which is what certification requires. This is "
            "average case - whether randomly drawn errors do, which is what usually "
            "happens. An ordering can be robust in practice and still uncertifiable, and "
            "the honest statement names which of the two it is."
        ),
    }
