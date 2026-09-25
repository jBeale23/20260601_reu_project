"""How many methods this benchmark can place in a certified order at all.

Every comparison in :mod:`validation.report` ranks methods on 129 curated labels. That
ranking is only meaningful if the labels can support it, and label noise puts a hard
ceiling on how fine a distinction any amount of analysis can recover.

The bound
---------
For a benchmark of ``n`` targets whose scores land on the grid of denominator ``n``, with
annotation error rate ``eps`` and ``delta`` the smallest difference worth calling a
difference::

    k(n, eps, delta) = n // (floor(c * n) + 1) + 1,    c = max(delta, 2 * eps)

is the largest number of methods that can be placed in a certified order, *however many
are entered*. The factor of two is the two-sided price of label noise. With ``delta = 0``
the bound reduces to ``k <= ceil(1 / (2 * eps))`` regardless of ``n``, which is the part
that matters here: **collecting more proteins does not buy resolution the labels lack.**

This is an impossibility result rather than a power calculation. For two methods whose
measured scores are closer than the noise budget, one can construct two truths, each
consistent with every label the benchmark recorded, that order the pair oppositely. The
ordering is then not a function of the data, and no analysis recovers it - only better
labels do.

Applied here
------------
The comparison this project cares about is the layout classifier against the profile-HMM
baseline, measured at 0.9225 and 0.9612. That gap of 0.0387 is resolvable only if the
subfamily labels carry under about 1.9% error. UniProt subfamily names are assigned by
curators partly from domain architecture, and for reference the CAID3 disorder benchmark
measures its own annotation error at 8.0%. So the report states the error rate at which
each comparison stops being decidable, rather than presenting an ordering whose validity
depends on an unmeasured quantity.

The bound and its Lean 4 proof are from DisorderNet (T. Marena, unpublished); this module
transcribes the statement and applies it to this benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Annotation error rates to report the benchmark's capacity at. The project has not
# measured its own, so a range is given rather than one assumed value; 0.08 is CAID3's
# measured rate for a comparable curated resource.
REFERENCE_ERROR_RATES = (0.01, 0.02, 0.05, 0.08, 0.10, 0.15)


def benchmark_capacity(n_targets: int, error_rate: float, *, delta: float = 0.0) -> int:
    """Largest number of methods a benchmark of this size and noise can certifiably order."""
    scale = max(delta, 2.0 * error_rate)
    if n_targets <= 0:
        return 1
    if scale <= 0:
        return n_targets
    return n_targets // (math.floor(scale * n_targets) + 1) + 1


def capacity_ceiling(error_rate: float, *, delta: float = 0.0) -> int:
    """The size-free ceiling: more targets cannot buy resolution the labels lack."""
    scale = max(delta, 2.0 * error_rate)
    return max(1, math.ceil(1.0 / scale)) if scale > 0 else 0


def max_resolvable_error_rate(score_gap: float) -> float:
    """Label error above which a gap of this size stops being decidable."""
    return max(0.0, score_gap / 2.0)


@dataclass(frozen=True, slots=True)
class PairwiseDecidability:
    """Whether one measured gap survives a given amount of label noise."""

    better: str
    worse: str
    score_gap: float
    max_error_rate: float

    def is_decidable_at(self, error_rate: float) -> bool:
        """Whether the ordering holds at this annotation error rate."""
        return 2.0 * error_rate < self.score_gap

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "better": self.better,
            "worse": self.worse,
            "score_gap": round(self.score_gap, 4),
            "max_error_rate_for_a_decidable_ordering": round(self.max_error_rate, 4),
            "decidable_at": {f"{rate:.0%}": self.is_decidable_at(rate) for rate in REFERENCE_ERROR_RATES},
        }


def pairwise_decidability(
    evaluations: Sequence[Mapping[str, object]],
    *,
    metric: str = "accuracy",
) -> list[dict[str, object]]:
    """For every pair of methods, the label error at which their ordering dissolves.

    Sorted by gap so the closest - and therefore least defensible - comparisons appear
    first, which is where a reader's scepticism belongs.
    """
    scored = [(str(item["name"]), float(item[metric])) for item in evaluations if metric in item]
    pairs: list[PairwiseDecidability] = []
    for index, (first_name, first_score) in enumerate(scored):
        for second_name, second_score in scored[index + 1 :]:
            better, worse = (first_name, second_name) if first_score >= second_score else (second_name, first_name)
            gap = abs(first_score - second_score)
            pairs.append(
                PairwiseDecidability(
                    better=better,
                    worse=worse,
                    score_gap=gap,
                    max_error_rate=max_resolvable_error_rate(gap),
                ),
            )
    pairs.sort(key=lambda item: item.score_gap)
    return [item.to_json_dict() for item in pairs]


def capacity_report(
    n_targets: int,
    evaluations: Sequence[Mapping[str, object]],
    *,
    n_effective: int | None = None,
) -> dict[str, object]:
    """Assemble the capacity section.

    ``n_effective`` is the number of genuinely independent targets, which here is the count
    of ortholog groups rather than of proteins: 122 of the 129 labelled proteins sit in a
    cross-genus ortholog group, so the labels carry far less independent information than
    their count suggests.
    """
    effective = n_effective if n_effective is not None else n_targets
    return {
        "n_targets": n_targets,
        "n_effective_targets": effective,
        "n_methods_compared": len(evaluations),
        "capacity_by_error_rate": {
            f"{rate:.0%}": {
                "capacity_at_n": benchmark_capacity(n_targets, rate),
                "capacity_at_effective_n": benchmark_capacity(effective, rate),
                "size_free_ceiling": capacity_ceiling(rate),
                "sufficient_for_this_comparison": benchmark_capacity(effective, rate) >= len(evaluations),
            }
            for rate in REFERENCE_ERROR_RATES
        },
        "pairwise_decidability": pairwise_decidability(evaluations),
        "note": (
            "Capacity is the number of methods a benchmark can place in a certified order, "
            "whatever the analysis. With delta = 0 it is bounded by ceil(1 / (2 * eps)) "
            "regardless of how many targets are collected, so more proteins cannot buy "
            "resolution the labels lack. Bound and Lean 4 proof from DisorderNet "
            "(T. Marena, unpublished)."
        ),
    }


# The sharp per-pair test, and why it changes nothing
# ---------------------------------------------------
# `pairwise_decidability` above applies the bound in its crude form: a gap survives only if
# it exceeds `2 * eps`, which prices in an adversary free to put every mislabelled target
# wherever it does the most damage. The impossibility construction's own hypotheses look
# tighter. Writing `E_P` for the targets a method gets wrong, `unresolvable_pair` needs
#
#     errors(Q) < errors(P) + 2 * min(nu, |E_Q \ E_P|)
#
# to build a truth that reverses the pair - twice the count of targets where Q errs *and P
# does not*, capped by the error budget. Targets both methods already fail move both scores
# together and can never reverse an ordering, so charging the margin against all of `nu`
# looks wasteful.
#
# It is not, and this module exists to show that rather than assume it. The margin can never
# exceed the exploitable count, since
#
#     margin = |E_Q| - |E_P| = |E_Q \ E_P| - |E_P \ E_Q| <= |E_Q \ E_P|
#
# so if the budget covers the exploitable set the pair is undecidable under both forms, and
# if it does not the sharp condition reduces to `margin >= 2 * budget` - the crude bound
# exactly. The two are equivalent. What this buys is not a tighter verdict but the per-pair
# diagnostics behind it: how much of the margin rests on failures the winner does not share.


def exclusive_errors(
    truth: Mapping[str, str],
    predictions: Mapping[str, str],
    other: Mapping[str, str],
    keys: Sequence[str],
) -> int:
    """Targets ``predictions`` gets wrong that ``other`` gets right."""
    return sum(1 for key in keys if predictions.get(key) != truth[key] and other.get(key) == truth[key])


def sharp_decidability(
    truth: Mapping[str, str],
    first: Mapping[str, str],
    second: Mapping[str, str],
    *,
    names: tuple[str, str],
    error_rate: float,
) -> dict[str, object] | None:
    """Whether a pair's ordering survives, using the theorem's own hypothesis.

    Returns ``None`` when the two methods score identically on the shared targets, since
    there is then no ordering to defend.
    """
    keys = sorted(set(truth) & set(first) & set(second))
    if not keys:
        return None
    errors_first = sum(1 for key in keys if first.get(key) != truth[key])
    errors_second = sum(1 for key in keys if second.get(key) != truth[key])
    if errors_first == errors_second:
        return None

    budget = round(error_rate * len(keys))
    if errors_first < errors_second:
        better, worse = names
        margin = errors_second - errors_first
        # The adversary can only exploit targets the loser fails and the winner does not.
        exploitable = exclusive_errors(truth, second, first, keys)
    else:
        worse, better = names
        margin = errors_first - errors_second
        exploitable = exclusive_errors(truth, first, second, keys)

    reversible = 2 * min(budget, exploitable)
    return {
        "better": better,
        "worse": worse,
        "margin_targets": margin,
        "error_budget_targets": budget,
        "exploitable_targets": exploitable,
        "reversible_by": reversible,
        # The ordering holds when the margin cannot be closed by any admissible relabelling.
        "decidable": margin >= reversible,
        "n_shared_targets": len(keys),
    }


def sharp_decidability_report(
    truth: Mapping[str, str],
    predictions: Mapping[str, Mapping[str, str]],
    *,
    error_rate: float = 0.05,
) -> dict[str, object]:
    """The sharp test applied to every pair of methods."""
    names = sorted(predictions)
    pairs = [
        result
        for index, first in enumerate(names)
        for second in names[index + 1 :]
        if (
            result := sharp_decidability(
                truth,
                predictions[first],
                predictions[second],
                names=(first, second),
                error_rate=error_rate,
            )
        )
        is not None
    ]
    pairs.sort(key=lambda item: int(item["margin_targets"]))
    return {
        "assumed_error_rate": error_rate,
        "n_pairs": len(pairs),
        "n_decidable": sum(1 for item in pairs if item["decidable"]),
        "pairs": pairs,
        "note": (
            "The crude bound charges a pair's margin against 2 * eps * n, which assumes "
            "every mislabelled target can be placed where it reverses the pair. This "
            "charges it against 2 * min(nu, exploitable) instead, where exploitable counts "
            "only targets the losing method fails and the winning one does not - the "
            "hypothesis unresolvable_pair actually needs. Targets both methods already get "
            "wrong shift both scores together and cannot reverse anything."
        ),
    }
