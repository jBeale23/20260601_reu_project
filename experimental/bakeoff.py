"""Head-to-head evaluation: incumbent classifier against challengers, on identical data.

The rule this module exists to enforce is that a challenger replaces the incumbent only by
winning a measurement, never by seeming promising. So every model here is scored on the
same proteins, against the same labels, under the same cross-validation, and the verdict
is computed rather than asserted.

What counts as winning
----------------------
Not "higher accuracy". Accuracy on this label set is dominated by class C, so a model can
gain on it by learning the prior. Promotion requires all three of:

1. **Macro-F1 above the incumbent**, so the gain is not confined to the majority class.
2. **The gain exceeding its own uncertainty** - the incumbent's accuracy must fall outside
   the challenger's Wilson 95% interval. On 129 labelled proteins that interval is roughly
   eight points wide, so a two-point lead is not a result.
3. **Beating the profile-HMM baseline**, since a model that loses to ``hmmsearch`` has not
   earned any complexity at all.

A challenger that fails any of these is reported with its numbers and left where it is.
That is a normal outcome and the report says so plainly rather than burying it.

Cross-validation is stratified and out-of-fold throughout: the challenger is fitted on the
training folds only. Fitting on everything and scoring on everything would let the model
memorise its own evaluation set, which is exactly the failure the incumbent's rule-based
design cannot have and a fitted model can.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from experimental.grammar_classifier import features_from_layout, fit_grammar_model
from experimental.structural_classifier import (
    EXCLUDE_MISSING,
    IMPUTE_MISSING,
    agreement_note,
    variants,
)
from validation.homology import _genus_blocked_folds, _stratified_folds
from validation.metrics import evaluate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.pipeline import ProteinLayout

logger = logging.getLogger(__name__)

NEURAL_NAME = "neural_hierarchical_cv"
CHALLENGER_NAME = "grammar_syntax_cv"

# The structural challengers, run alongside the annotation-free one rather than replacing
# it. Keeping all three separates "can grammar alone do this?" from "does structure add
# anything?" - questions that a single combined model would conflate.
STRUCTURAL_CHALLENGERS = (EXCLUDE_MISSING, IMPUTE_MISSING)
INCUMBENT_NAME = "domain_layout"

DEFAULT_N_FOLDS = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Verdict:
    """Whether a challenger earned promotion, and on what evidence."""

    challenger: str
    incumbent: str
    beats_macro_f1: bool
    beats_uncertainty: bool
    beats_homology_baseline: bool

    @property
    def promoted(self) -> bool:
        """Promotion requires every criterion, not a majority of them."""
        return self.beats_macro_f1 and self.beats_uncertainty and self.beats_homology_baseline

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "beats_macro_f1": self.beats_macro_f1,
            "gain_exceeds_uncertainty": self.beats_uncertainty,
            "beats_homology_baseline": self.beats_homology_baseline,
            "promoted": self.promoted,
            "decision": (
                f"{self.challenger} replaces {self.incumbent}"
                if self.promoted
                else f"{self.incumbent} retained; {self.challenger} did not clear every criterion"
            ),
        }


def grammar_predictions(
    layouts: Sequence[ProteinLayout],
    labels: Mapping[str, str],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    genus_blocked: bool = False,
) -> dict[str, str]:
    """Out-of-fold predictions from the grammar challenger.

    With ``genus_blocked`` every protein of a genus lands in the same fold, so no held-out
    protein has a same-genus relative in training. Both splits are reported: the gap
    between them measures how much of a score is memorised homology rather than learned
    grammar, and only the blocked number speaks to an unstudied organism.
    """
    by_accession = {layout.record.accession: layout for layout in layouts}
    usable = {accession: label for accession, label in labels.items() if accession in by_accession}
    if len(usable) < n_folds:
        return {}

    features = {accession: features_from_layout(by_accession[accession]) for accession in usable}
    if genus_blocked:
        organisms = {accession: by_accession[accession].record.organism_name for accession in usable}
        folds = _genus_blocked_folds(usable, organisms, n_folds=n_folds, seed=seed)
    else:
        folds = _stratified_folds(usable, n_folds=n_folds, seed=seed)

    predictions: dict[str, str] = {}
    for index, held_out in enumerate(folds):
        if not held_out:
            continue
        training = {
            accession: usable[accession]
            for position, fold in enumerate(folds)
            if position != index
            for accession in fold
        }
        model = fit_grammar_model(features, training)
        if model is None:
            continue
        for accession in held_out:
            predictions[accession] = model.predict(features[accession])
    return predictions


def predictions_from_features(
    features: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    genus_blocked_folds: Sequence[Sequence[str]] | None = None,
) -> dict[str, str]:
    """Out-of-fold predictions from any precomputed feature table.

    Shared by the structural challengers so they differ only in their features, never in
    how they are evaluated - otherwise a difference in score could come from either.
    """
    usable = {accession: label for accession, label in labels.items() if accession in features}
    if len(usable) < n_folds:
        return {}

    folds = list(genus_blocked_folds) if genus_blocked_folds else _stratified_folds(usable, n_folds=n_folds, seed=seed)
    folds = [[accession for accession in fold if accession in usable] for fold in folds]

    predictions: dict[str, str] = {}
    for index, held_out in enumerate(folds):
        if not held_out:
            continue
        training = {
            accession: usable[accession]
            for position, fold in enumerate(folds)
            if position != index
            for accession in fold
        }
        model = fit_grammar_model(features, training)
        if model is None:
            continue
        for accession in held_out:
            predictions[accession] = model.predict(features[accession])
    return predictions


def structural_predictions(
    layouts: Sequence[ProteinLayout],
    labels: Mapping[str, str],
    structural: Mapping[str, Mapping[str, float]],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Predictions from both structural variants, plus their coverage."""
    built = variants(layouts, structural)
    predictions: dict[str, dict[str, str]] = {}
    coverage: dict[str, object] = {}
    for name, (features, cover) in built.items():
        predictions[name] = predictions_from_features(features, labels, n_folds=n_folds, seed=seed)
        coverage[name] = cover.to_json_dict()
    return predictions, coverage


def _by_name(evaluations: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return next((item for item in evaluations if item.get("name") == name), None)


@dataclass(frozen=True, slots=True)
class BakeoffInputs:
    """Predictions from every model being compared, plus the labels to score against."""

    labels: Mapping[str, str]
    incumbent: Mapping[str, str]
    homology: Mapping[str, str]
    homology_genus_blocked: Mapping[str, str] = field(default_factory=dict)
    # Structural feature table, keyed by accession. Absent means the structural
    # challengers are skipped rather than run on nothing.
    structural: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # Out-of-fold predictions from the neural challenger, if a GPU run produced them.
    # Supplied rather than computed here: the neural model needs a GPU and tens of minutes
    # per seed, and every other model in this bake-off runs on CPU in seconds. Passing its
    # predictions in is what lets it be scored on exactly the same proteins as the rest,
    # which a separate report cannot do.
    neural: Mapping[str, str] = field(default_factory=dict)


def _score_all(
    truth: Mapping[str, str],
    inputs: BakeoffInputs,
    *,
    challenger: Mapping[str, str],
    blocked_challenger: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Score every non-structural model on the shared proteins.

    One place, so a model cannot be added to the comparison without being scored on exactly
    the same set as the rest - which is what stops a model gaining by declining hard cases.
    """
    named: list[tuple[str, Mapping[str, str]]] = [
        (INCUMBENT_NAME, inputs.incumbent),
        (CHALLENGER_NAME, challenger),
        ("profile_hmm_cv", inputs.homology),
        # Genus-blocked variants speak to an organism nothing related has been seen from.
        (f"{CHALLENGER_NAME}_genus_blocked", blocked_challenger),
        ("profile_hmm_genus_blocked", inputs.homology_genus_blocked),
        (NEURAL_NAME, inputs.neural),
    ]
    return [evaluate(name, truth, predictions).to_json_dict() for name, predictions in named if predictions]


def run_bakeoff(
    layouts: Sequence[ProteinLayout],
    inputs: BakeoffInputs,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
) -> dict[str, Any]:
    """Score the incumbent, the homology baseline, and the grammar challenger together."""
    labels = inputs.labels
    incumbent_predictions = inputs.incumbent
    homology_predictions = inputs.homology
    challenger = grammar_predictions(layouts, labels, n_folds=n_folds, seed=seed)
    blocked_challenger = grammar_predictions(
        layouts,
        labels,
        n_folds=n_folds,
        seed=seed,
        genus_blocked=True,
    )
    if not challenger:
        return {
            "challenger": CHALLENGER_NAME,
            "status": "not evaluated: too few labelled proteins for cross-validation",
        }

    # Every model is scored on exactly the proteins all of them made a prediction for, so
    # a model cannot gain by declining the hard cases.
    common = set(challenger) & set(incumbent_predictions) & set(labels)
    if homology_predictions:
        common &= set(homology_predictions)
    if inputs.neural:
        common &= set(inputs.neural)
    truth = {accession: labels[accession] for accession in sorted(common)}

    evaluations = _score_all(truth, inputs, challenger=challenger, blocked_challenger=blocked_challenger)

    # Both structural variants, scored on the same proteins as everything else.
    structural_coverage: dict[str, object] = {}
    structural_scores: dict[str, float] = {}
    if inputs.structural:
        structural_preds, structural_coverage = structural_predictions(
            layouts,
            labels,
            inputs.structural,
            n_folds=n_folds,
            seed=seed,
        )
        for name, preds in structural_preds.items():
            if not preds:
                continue
            result = evaluate(name, truth, preds).to_json_dict()
            evaluations.append(result)
            structural_scores[name] = float(result["accuracy"])

    incumbent = _by_name(evaluations, INCUMBENT_NAME)
    challenger_result = _by_name(evaluations, CHALLENGER_NAME)
    homology = _by_name(evaluations, "profile_hmm_cv")
    if incumbent is None or challenger_result is None:
        return {"challenger": CHALLENGER_NAME, "status": "not evaluated"}

    low = challenger_result["accuracy_95ci"][0]
    verdict = Verdict(
        challenger=CHALLENGER_NAME,
        incumbent=INCUMBENT_NAME,
        beats_macro_f1=challenger_result["macro_f1"] > incumbent["macro_f1"],
        # The incumbent must sit below the challenger's own interval, not merely below its
        # point estimate.
        beats_uncertainty=incumbent["accuracy"] < low,
        beats_homology_baseline=(homology is None or challenger_result["macro_f1"] > homology["macro_f1"]),
    )

    return {
        "challenger": CHALLENGER_NAME,
        "status": "evaluated",
        "n_evaluated": len(truth),
        "n_folds": n_folds,
        "seed": seed,
        "evaluations": evaluations,
        "verdict": verdict.to_json_dict(),
        "structural_coverage": structural_coverage,
        "structural_variant_agreement": (
            agreement_note(structural_scores[EXCLUDE_MISSING], structural_scores[IMPUTE_MISSING])
            if len(structural_scores) == len(STRUCTURAL_CHALLENGERS)
            else "structural challengers not run"
        ),
        "note": (
            "The challenger uses no InterPro annotation at any point, so its score is what "
            "would be available on a protein nothing has annotated. The incumbent's score "
            "is not, which is the comparison worth making rather than the raw gap."
        ),
    }


def class_confusion(labels: Mapping[str, str], predictions: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Actual-versus-predicted counts, for reading where a challenger actually fails."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for accession, truth in labels.items():
        predicted = predictions.get(accession)
        if predicted is not None:
            matrix[truth][predicted] += 1
    return {actual: dict(counts) for actual, counts in matrix.items()}
