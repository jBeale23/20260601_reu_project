"""Poisson-Boltzmann solvation energies, read against the domain-layout grammar.

The formal-charge sums in ``structure_analysis.features`` count charges; they cannot say
what those charges cost in water. APBS gives the polar solvation free energy, which is the
term an affinity calculation actually needs, and this module asks whether it varies with
the layout classes the grammar assigns.

One control dominates the design. Solvation energy is an extensive quantity: it scales with
how much protein there is, so the largest subclass will always look the most negative and a
raw comparison across subclasses measures length and nothing else. Every comparison here is
therefore run twice - on the raw energy and on energy per residue - and a rank correlation
against length is reported alongside, so a reader can see immediately whether a difference
survives the size it is confounded with. Where the two disagree, the per-residue result is
the one that means anything about chemistry.

The energies come from ``scripts/cluster/run_apbs_jdp.sh``: linearised PB, pdie 1.0,
sdie 78.54, srad 1.4, 298.15 K, on a per-molecule ``mg-auto`` grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.partner_systems import _cliffs_delta, _mann_whitney_p
from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

# Below this a group's median is too unstable to compare; the smallest stratum in the
# stratified APBS target set is 396, so this only ever excludes accidental fragments.
MIN_GROUP = 20

# Cliff's delta thresholds, the conventional small/medium/large cut points.
NEGLIGIBLE, SMALL, MEDIUM, LARGE = 0.147, 0.33, 0.474, 1.0

# Two-sided alpha, applied to Benjamini-Hochberg adjusted p-values.
SIGNIFICANCE_LEVEL = 0.05

# Fewest paired observations a rank correlation is meaningful over.
MIN_FOR_CORRELATION = 3

# Above this, raw energy is judged to be tracking length rather than chemistry.
STRONG_CORRELATION = 0.5

# accession, solvation, reference - the columns the APBS scraper writes.
ENERGY_COLUMNS = 3


@dataclass(frozen=True)
class SolvationEnergy:
    """One protein's Poisson-Boltzmann result, in kJ/mol."""

    accession: str
    solvation: float
    """Polar solvation free energy: the solvated calculation minus the reference."""
    reference: float
    """Vacuum self-energy. Retained because a wild value here flags a broken grid."""

    def per_residue(self, length: int) -> float | None:
        """Solvation normalised by chain length, or ``None`` when length is unusable."""
        if length <= 0:
            return None
        return self.solvation / length


def effect_label(delta: float) -> str:
    """Name a Cliff's delta by the conventional thresholds."""
    size = abs(delta)
    if size < NEGLIGIBLE:
        return "negligible"
    if size < SMALL:
        return "small"
    if size < MEDIUM:
        return "medium"
    return "large"


def load_energies(paths: Iterable[Path]) -> dict[str, SolvationEnergy]:
    """Read the APBS scraper's tab-separated output.

    Rows are ``accession, solvation, reference``. A malformed row is dropped with a warning
    rather than defaulted, because a zero energy would silently pull every group median
    toward zero and look like a real result.
    """
    energies: dict[str, SolvationEnergy] = {}
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            fields = line.split("\t")
            if len(fields) < ENERGY_COLUMNS or fields[0].lower().startswith("protein"):
                continue
            try:
                solvation, reference = float(fields[1]), float(fields[2])
            except ValueError:
                logger.warning("%s:%d: unparseable energy, dropping row.", path.name, number)
                continue
            if not (math.isfinite(solvation) and math.isfinite(reference)):
                logger.warning("%s:%d: non-finite energy, dropping row.", path.name, number)
                continue
            energies[fields[0]] = SolvationEnergy(fields[0], solvation, reference)
    return energies


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rho, with midranks for ties."""
    if len(first) != len(second) or len(first) < MIN_FOR_CORRELATION:
        return 0.0

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                ranks[order[position]] = shared
            index = stop + 1
        return ranks

    x, y = rank(first), rank(second)
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    var_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class GroupComparison:
    """One subclass against all others, on one measure."""

    group: str
    measure: str
    n_in: int
    n_out: int
    median_in: float
    median_out: float
    delta: float
    p_value: float
    p_adjusted: float = 1.0

    @property
    def significant(self) -> bool:
        """Significant after correction *and* past a negligible effect.

        Both conditions are required. With thousands of proteins a trivial difference
        reaches any p-value threshold, so significance alone would report noise as a
        finding.
        """
        return self.p_adjusted < SIGNIFICANCE_LEVEL and abs(self.delta) >= NEGLIGIBLE

    def to_json_dict(self) -> dict[str, object]:
        """Render for the JSON report."""
        return {
            "group": self.group,
            "measure": self.measure,
            "n_in_group": self.n_in,
            "n_other": self.n_out,
            "median_in_group": round(self.median_in, 4),
            "median_other": round(self.median_out, 4),
            "cliffs_delta": round(self.delta, 4),
            "effect": effect_label(self.delta),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "significant": self.significant,
        }


def compare_groups(
    values: Mapping[str, float],
    groups: Mapping[str, str],
    measure: str,
    *,
    min_group: int = MIN_GROUP,
) -> list[GroupComparison]:
    """Test each group against the pooled remainder, correcting across groups.

    One-against-the-rest rather than all pairs: the question is whether a layout class has
    a distinctive electrostatic character, not which of two classes is larger.
    """
    by_group: dict[str, list[float]] = {}
    for accession, value in values.items():
        group = groups.get(accession)
        if group:
            by_group.setdefault(group, []).append(value)

    comparisons: list[GroupComparison] = []
    for group, inside in sorted(by_group.items()):
        outside = [value for other, values_ in by_group.items() if other != group for value in values_]
        if len(inside) < min_group or len(outside) < min_group:
            continue
        comparisons.append(
            GroupComparison(
                group=group,
                measure=measure,
                n_in=len(inside),
                n_out=len(outside),
                median_in=_median(inside),
                median_out=_median(outside),
                delta=_cliffs_delta(inside, outside),
                p_value=_mann_whitney_p(inside, outside),
            )
        )

    adjusted = benjamini_hochberg({c.group: c.p_value for c in comparisons})
    return [
        GroupComparison(
            group=c.group,
            measure=c.measure,
            n_in=c.n_in,
            n_out=c.n_out,
            median_in=c.median_in,
            median_out=c.median_out,
            delta=c.delta,
            p_value=c.p_value,
            p_adjusted=adjusted.get(c.group, 1.0),
        )
        for c in comparisons
    ]


def electrostatics_report(
    energies: Mapping[str, SolvationEnergy],
    lengths: Mapping[str, int],
    groups: Mapping[str, str],
    *,
    min_group: int = MIN_GROUP,
) -> dict[str, object]:
    """Compare solvation energy across layout classes, raw and per residue.

    Args:
        energies: APBS results by accession.
        lengths: Chain length by accession, for the size control.
        groups: Layout class by accession.
        min_group: Smallest group worth testing.

    Returns:
        A JSON-ready report. ``size_confound`` carries the correlation between raw energy
        and length; when that is strong, only ``per_residue`` comparisons are informative,
        and ``interpretation`` says so rather than leaving it to the reader.
    """
    shared = [a for a in energies if a in lengths and lengths[a] > 0]
    raw = {a: energies[a].solvation for a in shared}
    per_residue = {a: energies[a].solvation / lengths[a] for a in shared}

    length_values = [float(lengths[a]) for a in shared]
    rho_raw = _spearman([raw[a] for a in shared], length_values)
    rho_norm = _spearman([per_residue[a] for a in shared], length_values)

    raw_tests = compare_groups(raw, groups, "solvation_kj_mol", min_group=min_group)
    norm_tests = compare_groups(per_residue, groups, "solvation_per_residue", min_group=min_group)

    survivors = [c for c in norm_tests if c.significant]
    raw_only = {c.group for c in raw_tests if c.significant} - {c.group for c in norm_tests if c.significant}

    if abs(rho_raw) > abs(rho_norm) and abs(rho_raw) > STRONG_CORRELATION:
        note = (
            f"Raw solvation energy tracks chain length (rho={rho_raw:.2f}), as an extensive "
            f"quantity must. Only the per-residue comparisons speak to chemistry."
        )
    else:
        note = f"Raw solvation energy is not dominated by length here (rho={rho_raw:.2f})."
    if raw_only:
        note += (
            f" {len(raw_only)} class(es) significant on raw energy lose that significance per "
            f"residue ({', '.join(sorted(raw_only))}), which marks them as size effects."
        )

    return {
        "n_proteins": len(shared),
        "n_energies_loaded": len(energies),
        "n_without_length": len(energies) - len(shared),
        "size_confound": {
            "spearman_raw_vs_length": round(rho_raw, 4),
            "spearman_per_residue_vs_length": round(rho_norm, 4),
        },
        "raw": [c.to_json_dict() for c in raw_tests],
        "per_residue": [c.to_json_dict() for c in norm_tests],
        "n_significant_per_residue": len(survivors),
        "interpretation": note,
    }


@dataclass(frozen=True)
class StratumSettings:
    """How finely to bin the covariate, and how small a group may get inside a bin."""

    n_strata: int = 4
    min_group: int = MIN_GROUP


DEFAULT_STRATA = StratumSettings()


def stratified_by_covariate(
    values: Mapping[str, float],
    groups: Mapping[str, str],
    covariate: Mapping[str, float],
    measure: str,
    *,
    settings: StratumSettings = DEFAULT_STRATA,
) -> dict[str, object]:
    """Repeat the group comparison inside quantile bins of a covariate.

    Solvation density per residue correlates with disorder, because intrinsically
    disordered regions are charge-enriched and hydrophobic-depleted. That raises the
    obvious objection to any class difference found in it: the classes differ in how
    disordered they are, so the electrostatics may be reporting disorder and nothing more.

    Binning on the covariate and re-testing within each bin answers that directly. A class
    whose effect holds in every bin, at a consistent sign, is not a proxy for the covariate;
    one that appears in a single bin is. Stratification rather than a partial correlation
    because the relationship need not be monotone, and because a per-bin table shows *where*
    an effect lives instead of averaging it away.

    Args:
        values: The measure under test, by accession.
        groups: Class label by accession.
        covariate: The variable to control for, by accession.
        measure: Name recorded on each comparison.
        settings: Bin count and the smallest group testable inside a bin.

    Returns:
        Per-stratum comparisons plus, per class, the bins it was significant in and whether
        its sign ever flipped. ``consistent`` classes are the ones the covariate does not
        explain.
    """
    n_strata, min_group = settings.n_strata, settings.min_group
    shared = sorted(a for a in values if a in covariate and a in groups)
    if len(shared) < n_strata * min_group:
        return {"n_strata": n_strata, "strata": [], "by_group": {}, "n_proteins": len(shared)}

    ordered = sorted(covariate[a] for a in shared)
    cuts = [ordered[min(len(ordered) - 1, int(len(ordered) * i / n_strata))] for i in range(1, n_strata)]

    def stratum_of(value: float) -> int:
        return sum(1 for cut in cuts if value > cut)

    strata: list[dict[str, object]] = []
    hits: dict[str, list[int]] = {}
    signs: dict[str, set[str]] = {}
    for index in range(n_strata):
        members = [a for a in shared if stratum_of(covariate[a]) == index]
        if not members:
            continue
        inside = {a: values[a] for a in members}
        labels = {a: groups[a] for a in members}
        comparisons = compare_groups(inside, labels, measure, min_group=min_group)
        covariate_values = [covariate[a] for a in members]
        strata.append(
            {
                "stratum": index + 1,
                "n": len(members),
                "covariate_min": round(min(covariate_values), 4),
                "covariate_max": round(max(covariate_values), 4),
                "comparisons": [c.to_json_dict() for c in comparisons],
            }
        )
        for comparison in comparisons:
            if comparison.significant:
                hits.setdefault(comparison.group, []).append(index + 1)
                signs.setdefault(comparison.group, set()).add("-" if comparison.delta < 0 else "+")

    by_group = {
        group: {
            "significant_in_strata": bins,
            "n_strata_significant": len(bins),
            "sign_stable": len(signs.get(group, set())) == 1,
            "consistent": len(bins) == len(strata) and len(signs.get(group, set())) == 1,
        }
        for group, bins in sorted(hits.items())
    }
    return {
        "n_proteins": len(shared),
        "n_strata": len(strata),
        "quantile_cuts": [round(c, 4) for c in cuts],
        "strata": strata,
        "by_group": by_group,
        "consistent_groups": sorted(g for g, v in by_group.items() if v["consistent"]),
    }
