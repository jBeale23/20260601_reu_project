"""What is the sequence grammar actually measuring?

The grammar model is the only one that improved when the rare architectures were added -
0.9070 to 0.9290, while the rule system fell to 0.8698 and the profile HMM collapsed to
0.5621. That is worth taking seriously, and it raises an obvious question: what is it
reading? A model that generalises where signature-matching fails is either seeing something
real or exploiting an artefact, and the two look identical from an accuracy table.

So this module correlates each grammar feature against everything else measured about the
same protein - structure, disorder, architecture, partner behaviour, variant burden - and
reports where the associations are.

On the word "causative"
-----------------------
Nothing here can establish causation, and the module is named to avoid implying otherwise.
Grammar features are computed *from* the sequence; so is everything they might explain. A
correlation between low-complexity grammar and disorder is close to a tautology, since both
measure compositional bias. Correlations that survive controlling for the obvious
confounders - length, disorder fraction, amino-acid composition - are the interesting ones,
and even those are associations. Causation would need perturbation: mutate the grammar
while holding composition fixed, and measure whether function changes. That is a wet-lab
experiment, not an analysis.

What is controlled, and why
---------------------------
Protein length confounds nearly everything here: longer proteins have more domains, more
partners, more variants, and different compressibility. Disorder fraction confounds every
compositional measure. Both are partialled out, and the raw and partial correlations are
reported side by side so the reader can see how much of an association survives.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Below this many paired observations a correlation is not worth quoting.
MIN_PAIRED_OBSERVATIONS = 20

SIGNIFICANCE_LEVEL = 0.05

# Confounders partialled out of every correlation. Length drives domain count, partner
# count and variant count alike; disorder drives every compositional measure.
DEFAULT_CONTROLS = ("n_residues", "fraction_disordered_plddt")

# Below this, the partialling denominator is numerically singular: the control explains
# essentially all of one variable, so no residual association can be recovered.
_SINGULAR_TOLERANCE = 1e-6


def _rank(values: Sequence[float]) -> list[float]:
    """Fractional ranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation.

    Rank-based rather than Pearson because almost nothing here is linear or normal: order
    scores cluster near zero, variant counts are heavily skewed, and pocket volumes have a
    long tail.
    """
    if len(first) != len(second) or len(first) < 2:  # noqa: PLR2004 - two points define nothing
        return 0.0
    x, y = _rank(first), _rank(second)
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y),
    )
    return numerator / denominator if denominator else 0.0


def partial_spearman(
    first: Sequence[float],
    second: Sequence[float],
    controls: Sequence[Sequence[float]],
) -> float:
    """Spearman correlation with confounders removed.

    Computed by first-order partialling, applied iteratively. Exact only for one control;
    with several it is an approximation, which is stated rather than hidden because the
    alternative - a full regression - would need a dependency this project does not carry
    and would not change which associations survive.
    """
    value = spearman(first, second)
    for control in controls:
        r_xz = spearman(first, control)
        r_yz = spearman(second, control)
        denominator = math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        # A control that is (near-)perfectly correlated with either variable leaves nothing
        # to partial: the denominator goes to zero and the ratio becomes arbitrary. Report
        # no residual association rather than whatever the division happened to produce -
        # a variable fully explained by a confounder has no independent signal by
        # construction, and a spurious 0.36 from a singular divide is worse than a zero.
        if denominator < _SINGULAR_TOLERANCE:
            return 0.0
        value = (value - r_xz * r_yz) / denominator
    return max(-1.0, min(1.0, value))


def _spearman_p(rho: float, n: int) -> float:
    """Two-sided p-value for a rank correlation, by the t approximation."""
    if n <= MIN_PAIRED_OBSERVATIONS or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else 1.0
    t = abs(rho) * math.sqrt((n - 2) / max(1e-12, 1 - rho**2))
    # Normal approximation to the t distribution; adequate at these sample sizes and it
    # avoids carrying a special-function dependency for a screening statistic.
    return math.erfc(t / math.sqrt(2.0))


@dataclass(frozen=True, slots=True)
class Correlate:
    """One grammar feature against one other measurement."""

    grammar_feature: str
    against: str
    n: int
    rho: float
    partial_rho: float
    p_value: float
    p_adjusted: float = 1.0

    @property
    def survives_controls(self) -> bool:
        """Whether the association is still substantial once confounders are removed.

        Half the raw magnitude is the bar. An association that halves under controlling for
        length and disorder was largely those things.
        """
        return abs(self.partial_rho) >= abs(self.rho) / 2 and abs(self.partial_rho) >= 0.1  # noqa: PLR2004

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "grammar_feature": self.grammar_feature,
            "against": self.against,
            "n": self.n,
            "rho": round(self.rho, 4),
            "partial_rho": round(self.partial_rho, 4),
            "survives_controls": self.survives_controls,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
        }


def correlate_grammar(
    grammar: Mapping[str, Mapping[str, float]],
    other: Mapping[str, Mapping[str, float]],
    *,
    controls: Sequence[str] = DEFAULT_CONTROLS,
) -> list[Correlate]:
    """Every grammar feature against every other measurement, corrected together.

    Both mappings are keyed by accession. Controls are drawn from ``other`` and excluded
    from the features they control for, so a measure is never partialled against itself.
    """
    shared = sorted(set(grammar) & set(other))
    if len(shared) < MIN_PAIRED_OBSERVATIONS:
        logger.warning("only %s proteins measured on both sides; not correlating", len(shared))
        return []

    grammar_names = sorted({name for values in grammar.values() for name in values})
    other_names = sorted({name for values in other.values() for name in values})

    results: list[Correlate] = []
    for feature in grammar_names:
        for target in other_names:
            if target in controls:
                continue
            paired = [
                (grammar[a][feature], other[a][target], [other[a].get(c, 0.0) for c in controls])
                for a in shared
                if feature in grammar[a] and target in other[a]
            ]
            if len(paired) < MIN_PAIRED_OBSERVATIONS:
                continue
            xs = [item[0] for item in paired]
            ys = [item[1] for item in paired]
            control_columns = [[item[2][index] for item in paired] for index in range(len(controls))]
            rho = spearman(xs, ys)
            results.append(
                Correlate(
                    grammar_feature=feature,
                    against=target,
                    n=len(paired),
                    rho=rho,
                    partial_rho=partial_spearman(xs, ys, control_columns),
                    p_value=_spearman_p(rho, len(paired)),
                ),
            )

    adjusted = benjamini_hochberg({f"{r.grammar_feature}|{r.against}": r.p_value for r in results})
    return [
        Correlate(
            grammar_feature=r.grammar_feature,
            against=r.against,
            n=r.n,
            rho=r.rho,
            partial_rho=r.partial_rho,
            p_value=r.p_value,
            p_adjusted=adjusted[f"{r.grammar_feature}|{r.against}"],
        )
        for r in results
    ]


def grammar_correlate_report(
    grammar: Mapping[str, Mapping[str, float]],
    other: Mapping[str, Mapping[str, float]],
    *,
    controls: Sequence[str] = DEFAULT_CONTROLS,
    max_reported: int = 60,
) -> dict[str, object]:
    """What the grammar tracks, once length and disorder are accounted for."""
    results = correlate_grammar(grammar, other, controls=controls)
    surviving = [r for r in results if r.survives_controls and r.p_adjusted < SIGNIFICANCE_LEVEL]
    surviving.sort(key=lambda r: -abs(r.partial_rho))

    return {
        "n_proteins": len(set(grammar) & set(other)),
        "n_correlations_tested": len(results),
        "n_significant": sum(1 for r in results if r.p_adjusted < SIGNIFICANCE_LEVEL),
        "n_surviving_controls": len(surviving),
        "controls": list(controls),
        "strongest": [r.to_json_dict() for r in surviving[:max_reported]],
        "interpretation": (
            "Grammar features are computed from the sequence, and so is much of what they "
            "are correlated against, so a raw association can be close to tautological - "
            "low-complexity grammar and disorder both measure compositional bias. The "
            "partial correlation removes protein length and disorder fraction; an "
            "association that halves under that was largely those things. Nothing here is "
            "causal: that would need the grammar perturbed while composition is held fixed, "
            "which is an experiment rather than an analysis."
        ),
    }
