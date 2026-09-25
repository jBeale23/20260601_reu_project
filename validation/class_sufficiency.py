"""Are the established A/B/C classes sufficient, and is a finer split better?

Every accuracy figure in the bake-off answers a different question from the one that
matters. The gold labels are themselves the A/B/C subfamily, lifted from curated protein
names, so a model scoring 0.93 has shown that architecture *predicts* A/B/C - not that
A/B/C is inadequate. If anything a high score argues the opposite: an arbitrary partition
would not be predictable from domain content at all.

Showing a finer classification is better needs a different experiment. It needs an
*independent* readout - something not used to define either classification - and a
demonstration that the finer split explains that readout better.

The control that makes this a test rather than a formality
----------------------------------------------------------
Splitting three classes into six will improve almost any association, for the same reason
adding parameters improves almost any fit. So the comparison is not against A/B/C alone; it
is against **random refinements of A/B/C with the identical shape** - the same number of
subgroups per parent, of the same sizes. Only the labels move.

That control asks the question that matters: is this particular split of A into two better
than an arbitrary split of A into two of the same sizes? If it is not, the subclasses are
granularity without content, whatever their accuracy against a label derived from them.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 0

# Below this many proteins carrying a readout, the contingency table is too sparse.
MIN_PROTEINS = 30
SIGNIFICANCE_LEVEL = 0.05


def cramers_v(pairs: Sequence[tuple[str, str]]) -> float:
    """Cramer's V between two categorical labellings.

    Normalised for table size, which is why it can be compared across classifications of
    different granularity at all - though not perfectly, which is the whole reason for the
    shape-matched permutation control.
    """
    if not pairs:
        return 0.0
    rows = sorted({row for row, _ in pairs})
    columns = sorted({column for _, column in pairs})
    if len(rows) < 2 or len(columns) < 2:  # noqa: PLR2004 - a single category has no association
        return 0.0

    table: dict[tuple[str, str], int] = Counter(pairs)
    row_totals: Counter[str] = Counter(row for row, _ in pairs)
    column_totals: Counter[str] = Counter(column for _, column in pairs)
    total = len(pairs)

    chi2 = 0.0
    for row in rows:
        for column in columns:
            expected = row_totals[row] * column_totals[column] / total
            if expected > 0:
                observed = table.get((row, column), 0)
                chi2 += (observed - expected) ** 2 / expected
    smaller = min(len(rows), len(columns)) - 1
    return math.sqrt(chi2 / (total * smaller)) if smaller > 0 else 0.0


def cramers_v_corrected(pairs: Sequence[tuple[str, str]]) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013): the null bias removed, nothing more.

    Under independence chi-square still carries an expectation of (r-1)(c-1) purely from
    sampling, so the plain statistic reports a positive association between variables that
    have none, and reports *more* of it the more groups the partition has. This correction
    subtracts that expectation from phi-square and shrinks the table dimensions to match, so
    an independent table scores zero whatever its shape.

    What it does **not** do is make partitions of different granularity comparable when
    there is real signal. A refinement that adds no information still scores below its
    parent, under both forms: chi-square rises only by the noise term while the denominator
    rises by the full change in ``min(r,c)-1``. That is arguably correct for V, which asks
    how close an association is to the maximum its table shape allows - a refinement with
    its parent's information is further from that maximum. But it makes V the wrong
    instrument for asking whether a *combination* of features carries more information about
    a readout than a single feature, because combinations are always finer. Use
    ``validation.modality_information.mutual_information`` for that question: information is
    monotone under refinement, so a combination cannot score below its own parents.

    Returns:
        Corrected V in [0, 1], or 0.0 when the table is degenerate or the association does
        not exceed what independence would produce.
    """
    if not pairs:
        return 0.0
    rows = sorted({row for row, _ in pairs})
    columns = sorted({column for _, column in pairs})
    if len(rows) < 2 or len(columns) < 2:  # noqa: PLR2004 - a single category has no association
        return 0.0

    table: dict[tuple[str, str], int] = Counter(pairs)
    row_totals: Counter[str] = Counter(row for row, _ in pairs)
    column_totals: Counter[str] = Counter(column for _, column in pairs)
    total = len(pairs)

    chi2 = 0.0
    for row in rows:
        for column in columns:
            expected = row_totals[row] * column_totals[column] / total
            if expected > 0:
                observed = table.get((row, column), 0)
                chi2 += (observed - expected) ** 2 / expected

    n_rows, n_columns = len(rows), len(columns)
    if total <= 1:
        return 0.0

    phi2 = chi2 / total
    # Subtract the association independence alone would produce, then shrink the dimensions
    # by the same argument. Both terms are floored at zero: below the null there is no
    # association to report, and a negative under the square root is not a smaller effect.
    phi2_corrected = max(0.0, phi2 - (n_rows - 1) * (n_columns - 1) / (total - 1))
    rows_corrected = n_rows - (n_rows - 1) ** 2 / (total - 1)
    columns_corrected = n_columns - (n_columns - 1) ** 2 / (total - 1)
    smaller_corrected = min(rows_corrected, columns_corrected) - 1
    if smaller_corrected <= 0:
        return 0.0
    return math.sqrt(phi2_corrected / smaller_corrected)


def _shape_matched_shuffle(
    coarse: Mapping[str, str],
    fine: Mapping[str, str],
    rng: random.Random,
) -> dict[str, str]:
    """A random refinement of the coarse labels with the same shape as the fine one.

    Within each coarse class, the fine labels are permuted among its members. Subgroup
    counts and sizes are preserved exactly; only which protein carries which label changes.
    That isolates the content of the split from its granularity.
    """
    by_coarse: dict[str, list[str]] = defaultdict(list)
    for accession, label in coarse.items():
        if accession in fine:
            by_coarse[label].append(accession)

    shuffled: dict[str, str] = {}
    for members in by_coarse.values():
        labels = [fine[accession] for accession in members]
        rng.shuffle(labels)
        shuffled.update(dict(zip(members, labels, strict=True)))
    return shuffled


@dataclass(frozen=True, slots=True)
class SufficiencyResult:
    """Whether a finer classification explains a readout better than its parent."""

    readout: str
    n_proteins: int
    coarse_v: float
    fine_v: float
    null_mean_v: float
    null_max_v: float
    p_value: float
    n_permutations: int

    @property
    def improvement(self) -> float:
        """How much the finer split adds over its parent."""
        return self.fine_v - self.coarse_v

    @property
    def beats_random_refinement(self) -> bool:
        """Whether the split's content, not merely its granularity, explains the gain."""
        return self.p_value < SIGNIFICANCE_LEVEL

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "readout": self.readout,
            "n_proteins": self.n_proteins,
            "coarse_cramers_v": round(self.coarse_v, 4),
            "fine_cramers_v": round(self.fine_v, 4),
            "improvement": round(self.improvement, 4),
            "null_mean_cramers_v": round(self.null_mean_v, 4),
            "null_max_cramers_v": round(self.null_max_v, 4),
            "p_value": round(self.p_value, 5),
            "beats_random_refinement": self.beats_random_refinement,
            "n_permutations": self.n_permutations,
        }


def assess_sufficiency(  # noqa: PLR0913 - a comparison needs both classifications, the readout, and the null's settings
    coarse: Mapping[str, str],
    fine: Mapping[str, str],
    readout: Mapping[str, str],
    *,
    readout_name: str = "readout",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> SufficiencyResult | None:
    """Does the finer classification explain the readout better than chance allows?

    Args:
        coarse: The established classification - here A/B/C.
        fine: The proposed refinement.
        readout: An independent categorical observation per protein, used to define
            neither classification.
        readout_name: What the readout is, for the report.
        n_permutations: Shape-matched random refinements to compare against.
        seed: Seed, so the null is reproducible.

    Returns:
        The comparison, or ``None`` when too few proteins carry all three.
    """
    shared = sorted(set(coarse) & set(fine) & set(readout))
    if len(shared) < MIN_PROTEINS:
        logger.warning("only %s proteins carry class, subclass and %s", len(shared), readout_name)
        return None

    coarse_v = cramers_v([(coarse[a], readout[a]) for a in shared])
    fine_v = cramers_v([(fine[a], readout[a]) for a in shared])

    rng = random.Random(seed)  # noqa: S311 - a permutation null, not a secret
    null: list[float] = []
    restricted_coarse = {a: coarse[a] for a in shared}
    restricted_fine = {a: fine[a] for a in shared}
    for _ in range(n_permutations):
        shuffled = _shape_matched_shuffle(restricted_coarse, restricted_fine, rng)
        null.append(cramers_v([(shuffled[a], readout[a]) for a in shared]))

    # One-sided: the question is whether the real split beats an arbitrary one, and the
    # +1 is the standard correction that keeps a p-value of exactly zero off the page.
    at_least = sum(1 for value in null if value >= fine_v)
    return SufficiencyResult(
        readout=readout_name,
        n_proteins=len(shared),
        coarse_v=coarse_v,
        fine_v=fine_v,
        null_mean_v=sum(null) / len(null) if null else 0.0,
        null_max_v=max(null) if null else 0.0,
        p_value=(at_least + 1) / (n_permutations + 1),
        n_permutations=n_permutations,
    )


def sufficiency_report(
    coarse: Mapping[str, str],
    fine: Mapping[str, str],
    readouts: Mapping[str, Mapping[str, str]],
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Test the refinement against every independent readout available."""
    results = [
        result
        for name, readout in sorted(readouts.items())
        if (
            result := assess_sufficiency(
                coarse,
                fine,
                readout,
                readout_name=name,
                n_permutations=n_permutations,
                seed=seed,
            )
        )
        is not None
    ]
    return {
        "n_coarse_classes": len(set(coarse.values())),
        "n_fine_classes": len(set(fine.values())),
        "readouts_tested": [item.readout for item in results],
        "n_readouts_where_split_beats_random": sum(1 for item in results if item.beats_random_refinement),
        "results": [item.to_json_dict() for item in results],
        "interpretation": (
            "The bake-off's accuracy figures cannot answer this: the gold labels are the "
            "A/B/C subfamily itself, so a high score shows architecture predicts A/B/C "
            "rather than that A/B/C is inadequate. This compares both classifications "
            "against a readout used to define neither, and against random refinements of "
            "A/B/C with identical shape - the same subgroups, the same sizes, only the "
            "labels moved. Splitting three classes into six improves almost any "
            "association for the same reason adding parameters improves almost any fit; "
            "beating the shape-matched null is what distinguishes content from granularity."
        ),
    }
