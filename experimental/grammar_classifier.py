"""Challenger: classify JDPs from hierarchical n-mer grammar rather than architecture.

The incumbent classifier in :mod:`domain_layout.profiles` reasons over *architecture* -
which domains are present, in what order, with what between them. That works precisely
when the domains are annotated, which is the case for well-studied organisms and
increasingly not the case as one moves away from them.

This challenger asks whether the same classes are recoverable from sequence *syntax*
alone: reduced-alphabet n-mer statistics, entropy at several orders, and compression-based
order scores, computed over the whole protein and over its unalignable regions, with no
InterPro annotation used at any point. If it works, it transfers to proteins InterPro
cannot annotate. If it does not, that is worth knowing precisely and is the expected
outcome for a first attempt.

The model is deliberately simple - a multinomial-style linear scorer over grammar features,
fitted by class-conditional means with shared variance, which is Gaussian naive Bayes in
all but name. A simple model is the right challenger: if a linear function of twenty
grammar features matches an architecture-aware rule system, that is a strong result about
the *features*. Reaching for a deep network first would leave it unclear whether any gain
came from the representation or from capacity.

Nothing here is imported by :mod:`domain_layout`. Replacing the incumbent is a decision
for the bake-off in :mod:`experimental.bakeoff`, not for this module.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.constants import ROUTE_SHARK
from domain_layout.grammar import (
    DEFAULT_MAX_ORDER,
    REDUCED_ALPHABETS,
    block_entropies,
    charge_profile,
    conditional_complexity_profile,
    nmer_counts,
    order_score,
    profile_sequence,
    reduce_sequence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.pipeline import ProteinLayout

# Alphabets whose grammars are combined into one feature vector. Three different
# reductions see different structure: chemical7 carries the most information, charge3
# isolates the electrostatic patterning that distinguishes JDP linkers, and hydrophobic2
# is coarse enough to stay estimable on short regions.
FEATURE_ALPHABETS = ("chemical7", "charge3", "hydrophobic2")

# Variance floor, so a feature that happens to be constant within a class cannot produce
# an infinite log-likelihood and dominate every prediction.
_MIN_VARIANCE = 1e-6

# A class needs at least this many examples before a variance can be estimated from it,
# and at least this many classes must survive for the result to be a classifier at all.
MIN_EXAMPLES_PER_CLASS = 2
MIN_CLASSES = 2


def _region_sequences(layout: ProteinLayout) -> list[str]:
    """The unalignable regions, which is where grammar carries the most signal."""
    return [region.sequence for region in layout.regions if region.route == ROUTE_SHARK and region.sequence]


def grammar_features(sequence: str, regions: Sequence[str] = ()) -> dict[str, float]:
    """Build the grammar feature vector for one protein.

    No InterPro annotation is consulted anywhere in this function. That is the whole
    point: the features must be computable for a protein nothing has ever annotated.
    """
    features: dict[str, float] = {}

    for alphabet in FEATURE_ALPHABETS:
        reduced = reduce_sequence(sequence, alphabet=alphabet)
        entropies = block_entropies(reduced, max_order=DEFAULT_MAX_ORDER)
        for order, value in enumerate(entropies, start=1):
            features[f"{alphabet}_H{order}"] = value
        # Entropy rates: what each extra residue of context still fails to explain.
        previous = 0.0
        for order, value in enumerate(entropies, start=1):
            features[f"{alphabet}_h{order}"] = max(0.0, value - previous)
            previous = value

        counts = nmer_counts(reduced, 1)
        total = sum(counts.values()) or 1
        for symbol in sorted(set(REDUCED_ALPHABETS[alphabet].values())):
            features[f"{alphabet}_f_{symbol}"] = counts.get(symbol, 0) / total

    whole = profile_sequence(sequence)
    features["order_score"] = whole.order_score
    features["compression_ratio"] = whole.compression_ratio

    # Conditional complexity along the sequence, normalised twice so absolute length
    # cannot enter: bits *per residue* rather than per protein, and the peak expressed
    # against the sequence's own mean.
    #
    # Raw length was previously a feature here, and it was a leak: class C is largely
    # "J-domain only" and therefore shorter, so the model could score well by learning
    # length rather than grammar. These replace it with quantities a 90-residue protein
    # and a 900-residue one can be compared on.
    profile = conditional_complexity_profile(sequence)
    features["conditional_mean_bits"] = profile.mean
    features["conditional_spread"] = profile.spread
    features["conditional_dynamic_range"] = profile.dynamic_range
    features["conditional_trough"] = profile.trough

    # Charge, locally and globally. Net charge over a whole protein averages a basic patch
    # against an acidic one and reports zero for a strongly polarised sequence, so the
    # windowed terms carry information the global one cannot.
    charge = charge_profile(sequence)
    features["net_charge_per_residue"] = charge.net_charge_per_residue
    features["fraction_charged"] = charge.fraction_charged
    features["charge_segregation"] = charge.charge_segregation
    features["max_window_charge"] = charge.max_window_charge
    features["min_window_charge"] = charge.min_window_charge

    # Region-level grammar, summarised. The count and the coverage fraction are structural
    # facts about the protein that need no annotation to compute.
    # Region-level grammar. Coverage is a *fraction*, and the region count is expressed
    # per hundred residues rather than absolutely, so neither smuggles length back in.
    if regions:
        region_scores = [order_score(reduce_sequence(region)) for region in regions]
        features["region_density"] = len(regions) / (max(1, len(sequence)) / 100.0)
        features["region_order_mean"] = sum(region_scores) / len(region_scores)
        features["region_order_max"] = max(region_scores)
        features["region_coverage"] = sum(len(region) for region in regions) / max(1, len(sequence))
    else:
        features["region_density"] = 0.0
        features["region_order_mean"] = 0.0
        features["region_order_max"] = 0.0
        features["region_coverage"] = 0.0
    return features


def features_from_layout(layout: ProteinLayout) -> dict[str, float]:
    """Grammar features for one analysed protein.

    Takes a layout only to reach the sequence and the region boundaries; no class, no
    evidence flag, and no domain annotation from the layout is read, so the incumbent's
    reasoning cannot leak into the challenger's features.
    """
    return grammar_features(layout.record.sequence, _region_sequences(layout))


@dataclass(frozen=True, slots=True)
class GrammarModel:
    """A fitted class-conditional Gaussian model over grammar features."""

    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    means: dict[str, tuple[float, ...]]
    variances: dict[str, tuple[float, ...]]
    log_priors: dict[str, float]

    def predict(self, features: Mapping[str, float]) -> str:
        """Return the highest-posterior class."""
        return max(self.classes, key=lambda label: self.log_likelihood(features, label))

    def log_likelihood(self, features: Mapping[str, float], label: str) -> float:
        """Unnormalised log posterior of one class."""
        total = self.log_priors[label]
        means = self.means[label]
        variances = self.variances[label]
        for index, name in enumerate(self.feature_names):
            value = features.get(name, 0.0)
            variance = variances[index]
            total += -0.5 * (math.log(2 * math.pi * variance) + ((value - means[index]) ** 2) / variance)
        return total

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the model's shape for the report."""
        return {
            "classes": list(self.classes),
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
        }


def fit_grammar_model(
    features_by_accession: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
) -> GrammarModel | None:
    """Fit class-conditional means and variances over the labelled proteins.

    Returns ``None`` when fewer than two classes carry usable examples, since a classifier
    with one class is not a classifier.
    """
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for accession, label in labels.items():
        features = features_by_accession.get(accession)
        if features:
            grouped[label].append(features)

    usable = {label: rows for label, rows in grouped.items() if len(rows) >= MIN_EXAMPLES_PER_CLASS}
    if len(usable) < MIN_CLASSES:
        return None

    feature_names = tuple(sorted({name for rows in usable.values() for row in rows for name in row}))
    total = sum(len(rows) for rows in usable.values())

    means: dict[str, tuple[float, ...]] = {}
    variances: dict[str, tuple[float, ...]] = {}
    log_priors: dict[str, float] = {}
    for label, rows in usable.items():
        column_means: list[float] = []
        column_variances: list[float] = []
        for name in feature_names:
            values = [row.get(name, 0.0) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            column_means.append(mean)
            column_variances.append(max(_MIN_VARIANCE, variance))
        means[label] = tuple(column_means)
        variances[label] = tuple(column_variances)
        log_priors[label] = math.log(len(rows) / total)

    return GrammarModel(
        classes=tuple(sorted(usable)),
        feature_names=feature_names,
        means=means,
        variances=variances,
        log_priors=log_priors,
    )
