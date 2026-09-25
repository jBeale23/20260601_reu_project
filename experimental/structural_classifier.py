"""Challengers that add structure to grammar, in two honest variants.

The grammar challenger uses no annotation and no structure, which is what lets its score
stand for what would be available on a protein nothing has studied. Adding AlphaFold
features changes the question from "can grammar alone do this?" to "can grammar plus
structure do this?", and both are worth answering - so this is a *second* challenger rather
than a modification of the first.

Missing models force a choice, so both choices are run
------------------------------------------------------
AlphaFold covers 142,948 of 181,526 proteins here: 78.8%. The 38,578 without a model cannot
be scored on structure, and there is no answer to that which is simply correct.

**Excluding** them is honest about what was measured but shrinks the evaluation set and
biases it toward proteins the modelling pipeline handled well - which correlates with being
well studied, the same bias that already afflicts the labels.

**Imputing** them keeps every protein but invents values. Imputation is done with the
per-class median and paired with an explicit ``has_structure`` indicator, so the model can
learn to discount imputed rows rather than being misled into treating them as measurements.

Running both is the point. If they agree, missing models are not distorting the result and
either number can be quoted. If they disagree, the structural signal depends on *which*
proteins have models, which is a finding about model availability rather than about
biology - and one that would be invisible from either variant alone.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from experimental.grammar_classifier import features_from_layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from domain_layout.pipeline import ProteinLayout

logger = logging.getLogger(__name__)

EXCLUDE_MISSING = "grammar_structure_excluded"
IMPUTE_MISSING = "grammar_structure_imputed"

# Structural columns added to the grammar vector. Every one is per-residue, a fraction, or
# a ratio, so none reintroduces absolute length - the leak that made raw length unusable in
# the grammar challenger.
STRUCTURAL_FEATURES = (
    "mean_plddt",
    "fraction_disordered_plddt",
    "fraction_confident_plddt",
    "compactness",
    "relative_contact_order",
    "fraction_buried",
    "surface_charge_density",
    "mean_relative_sasa",
    "fraction_exposed",
    "n_pockets",
    "largest_pocket_mean_rsa",
)

# Absolute quantities that must be normalised before use, paired with the column that
# normalises them. Total surface area and pocket volume both scale with protein size.
_NORMALISED_FEATURES = {
    "total_sasa": "n_residues",
    "largest_pocket_volume": "n_residues",
    "exposed_net_charge": "n_residues",
    "buried_net_charge_sasa": "n_residues",
}


@dataclass(frozen=True, slots=True)
class StructuralCoverage:
    """How much of an evaluation set carries a structural model."""

    n_proteins: int
    n_with_structure: int

    @property
    def coverage(self) -> float:
        """Share carrying a model."""
        return self.n_with_structure / self.n_proteins if self.n_proteins else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "n_proteins": self.n_proteins,
            "n_with_structure": self.n_with_structure,
            "n_without_structure": self.n_proteins - self.n_with_structure,
            "structure_coverage": round(self.coverage, 4),
        }


def load_structural_features(path: Path) -> dict[str, dict[str, float]]:
    """Read the structural feature table, keyed by accession.

    Absolute quantities are divided by residue count on the way in, so a protein cannot
    score highly on surface area simply by being large.
    """
    features: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accession = row.get("accession")
            if not accession:
                continue
            values: dict[str, float] = {}
            for name in STRUCTURAL_FEATURES:
                raw = row.get(name, "")
                if raw not in ("", None):
                    try:
                        values[name] = float(raw)
                    except ValueError:
                        continue
            residues = float(row.get("n_residues") or 0.0)
            for name, denominator in _NORMALISED_FEATURES.items():
                raw = row.get(name, "")
                scale = float(row.get(denominator) or 0.0) if denominator else 1.0
                if raw not in ("", None) and scale > 0:
                    try:
                        values[f"{name}_per_residue"] = float(raw) / scale
                    except ValueError:
                        continue
            _ = residues
            if values:
                features[accession] = values
    return features


def _feature_names(structural: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    return tuple(sorted({name for values in structural.values() for name in values}))


def combined_features(
    layouts: Sequence[ProteinLayout],
    structural: Mapping[str, Mapping[str, float]],
    *,
    impute_missing: bool,
) -> tuple[dict[str, dict[str, float]], StructuralCoverage]:
    """Grammar features plus structure, for every protein that qualifies.

    With ``impute_missing`` false, proteins without a model are absent from the result -
    the evaluation simply does not include them. With it true, every protein is present and
    the missing structural values are filled with the column median, alongside a
    ``has_structure`` flag set to zero so the model is told which rows were invented.
    """
    names = _feature_names(structural)
    medians = {
        name: median([values[name] for values in structural.values() if name in values] or [0.0]) for name in names
    }

    combined: dict[str, dict[str, float]] = {}
    n_with = 0
    for layout in layouts:
        accession = layout.record.accession
        measured = structural.get(accession)
        if measured is None and not impute_missing:
            continue

        row = dict(features_from_layout(layout))
        if measured is not None:
            n_with += 1
            row.update({name: measured.get(name, medians[name]) for name in names})
            row["has_structure"] = 1.0
        else:
            row.update(medians)
            # The model is told this row was filled in, so it can learn to discount it
            # rather than treating an invented value as a measurement.
            row["has_structure"] = 0.0
        combined[accession] = row

    return combined, StructuralCoverage(n_proteins=len(combined), n_with_structure=n_with)


def variants(
    layouts: Sequence[ProteinLayout],
    structural: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[dict[str, dict[str, float]], StructuralCoverage]]:
    """Both structural challengers, so the effect of missing models is measurable.

    Returns each variant's features and its coverage. Comparing the two scores answers a
    question neither answers alone: whether the structural signal survives the proteins
    that have no model.
    """
    return {
        EXCLUDE_MISSING: combined_features(layouts, structural, impute_missing=False),
        IMPUTE_MISSING: combined_features(layouts, structural, impute_missing=True),
    }


def agreement_note(excluded_accuracy: float, imputed_accuracy: float, *, tolerance: float = 0.02) -> str:
    """Plain statement of what the difference between the two variants means."""
    gap = abs(excluded_accuracy - imputed_accuracy)
    if gap <= tolerance:
        return (
            f"The two variants agree to within {gap:.3f}, so proteins without a model are "
            f"not distorting the structural result and either number can be quoted."
        )
    return (
        f"The two variants differ by {gap:.3f}, so the structural signal depends on which "
        f"proteins have models. That is a finding about model availability - which tracks "
        f"how well studied a protein is - rather than about biology, and neither number "
        f"should be quoted without the other."
    )
