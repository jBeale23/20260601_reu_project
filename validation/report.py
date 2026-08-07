"""Assemble the validation report: labels, quality, recurrence, and nulls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from domain_layout.constants import NOVELTY_THRESHOLD
from domain_layout.pipeline import LayoutRunConfig, analyze_store
from jdp_classifier.architecture import features_from_ida
from jdp_classifier.domain_evidence import ida_from_record
from jdp_classifier.rules import predict_class
from validation.labels import collect_gold_labels
from validation.metrics import evaluate, majority_baseline
from validation.nulls import novelty_null, score_threshold_sweep
from validation.quality import assess_records, summarize_quality
from validation.recurrence import benjamini_hochberg, recurrence_with_null

if TYPE_CHECKING:
    from pathlib import Path

    from domain_layout.pipeline import ProteinLayout
    from domain_layout.profiles import ReferenceRegion
    from domain_layout.records import DomainStore

# Architectures seen in fewer proteins than this are not tested for taxonomic spread:
# with one or two members there is no spread to measure and the test has no power.
MIN_PROTEINS_FOR_RECURRENCE = 3

# Conventional false-discovery cut for reporting how many architectures are significant.
SIGNIFICANCE_FDR = 0.05


def _architecture_key(layout: ProteinLayout) -> str:
    return layout.evidence.domain_family_layout or "(no annotated domains)"


def _layout_predictions(layouts: list[ProteinLayout]) -> dict[str, str]:
    return {layout.record.accession: layout.classification.predicted_class for layout in layouts}


def _architecture_only_predictions(layouts: list[ProteinLayout]) -> dict[str, str]:
    """Baseline: the v1 rules applied to the Pfam architecture string alone."""
    return {
        layout.record.accession: predict_class(features_from_ida(ida_from_record(layout.record))) for layout in layouts
    }


@dataclass(frozen=True, slots=True)
class PermutationSettings:
    """How many permutations to run for each null, and the seed that fixes them."""

    novelty: int = 200
    recurrence: int = 1000
    seed: int = 0


DEFAULT_PERMUTATIONS = PermutationSettings()


def build_validation_report(
    store: DomainStore,
    *,
    references: list[ReferenceRegion] | None = None,
    config: LayoutRunConfig | None = None,
    permutations: PermutationSettings = DEFAULT_PERMUTATIONS,
) -> dict[str, Any]:
    """Run every validation analysis over a domain store and return a report."""
    run_config = config or LayoutRunConfig()
    result = analyze_store(store, config=run_config, references=references or [])
    layouts = result.layouts

    quality = assess_records(store)
    passing = {accession for accession, item in quality.items() if item.passes}
    complete_layouts = [layout for layout in layouts if layout.record.accession in passing]

    gold = collect_gold_labels(store)
    truth = {accession: label.label for accession, label in gold.items()}
    # Calibration is scored on the same quality-filtered set as everything else. A curated
    # entry that UniProt flags as a fragment still carries a subfamily name, and scoring
    # the classifier on residues that were never sequenced measures neither fairly.
    truth_complete = {accession: label for accession, label in truth.items() if accession in passing}

    predictions = _layout_predictions(complete_layouts)
    evaluated_accessions = sorted(set(truth_complete) & set(predictions))
    baselines = {
        "majority_class": majority_baseline(truth_complete, evaluated_accessions),
        "architecture_only": _architecture_only_predictions(complete_layouts),
    }

    evaluations = [evaluate("domain_layout", truth_complete, predictions).to_json_dict()]
    evaluations.extend(evaluate(name, truth_complete, values).to_json_dict() for name, values in baselines.items())
    # Also scored without the filter, so the filter cannot be accused of selecting the
    # examples that happen to make the classifier look good.
    unfiltered = evaluate("domain_layout_unfiltered", truth, _layout_predictions(layouts)).to_json_dict()

    architectures = {layout.record.accession: _architecture_key(layout) for layout in complete_layouts}
    organisms = {layout.record.accession: layout.record.organism_name for layout in complete_layouts}
    recurrence = recurrence_with_null(
        architectures,
        organisms,
        n_permutations=permutations.recurrence,
        seed=permutations.seed,
    )
    testable = {key: item for key, item in recurrence.items() if item.n_proteins >= MIN_PROTEINS_FOR_RECURRENCE}
    adjusted = benjamini_hochberg({key: item.p_value for key, item in testable.items()})

    evidence = [layout.evidence for layout in complete_layouts]
    matches = [layout.shark_match for layout in complete_layouts]
    null = novelty_null(evidence, matches, n_permutations=permutations.novelty, seed=permutations.seed)

    candidates_all = sum(1 for layout in layouts if layout.classification.novel_class_candidate)
    candidates_complete = sum(1 for layout in complete_layouts if layout.classification.novel_class_candidate)

    return {
        "dataset": {
            "n_proteins_in_store": len(store),
            "n_analyzed": len(layouts),
            "n_after_quality_filter": len(complete_layouts),
            "disorder_backend": result.disorder_backend,
            "shark_backend": result.shark_backend,
            "n_reference_regions": result.n_references,
        },
        # Recorded so a rerun reproduces every p-value in this report exactly.
        "permutation_settings": {
            "seed": permutations.seed,
            "novelty_permutations": permutations.novelty,
            "recurrence_permutations": permutations.recurrence,
        },
        "quality_control": summarize_quality(quality.values()),
        "label_calibration": {
            "label_source": "curated UniProt subfamily nomenclature (reviewed entries only)",
            "n_labelled": len(truth),
            "n_labelled_after_quality_filter": len(truth_complete),
            "n_evaluated": len(evaluated_accessions),
            "evaluations": evaluations,
            "unfiltered_comparison": unfiltered,
        },
        "novelty": {
            "threshold": NOVELTY_THRESHOLD,
            "candidates_before_quality_filter": candidates_all,
            "candidates_after_quality_filter": candidates_complete,
            "removed_by_quality_filter": candidates_all - candidates_complete,
            "permutation_null": null.to_json_dict(),
            "threshold_sensitivity": score_threshold_sweep(evidence, matches),
        },
        "recurrence": {
            "significance_fdr": SIGNIFICANCE_FDR,
            "min_proteins_tested": MIN_PROTEINS_FOR_RECURRENCE,
            "n_architectures": len(recurrence),
            "n_architectures_tested": len(testable),
            "n_significant_at_fdr": sum(1 for value in adjusted.values() if value < SIGNIFICANCE_FDR),
            "architectures": [
                {**item.to_json_dict(), "p_adjusted": round(adjusted.get(key, 1.0), 5)}
                for key, item in sorted(testable.items(), key=lambda kv: (-kv[1].n_genera, kv[0]))[:25]
            ],
        },
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the report as indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def format_summary(report: dict[str, Any]) -> str:
    """Render a short human-readable digest of the report."""
    dataset = report["dataset"]
    quality = report["quality_control"]
    labels = report["label_calibration"]
    novelty = report["novelty"]
    recurrence = report["recurrence"]

    lines = [
        f"proteins analyzed: {dataset['n_analyzed']} ({dataset['n_after_quality_filter']} pass quality control)",
        f"quality exclusions: {quality['n_excluded']} {quality['exclusion_reasons']}",
        "",
        f"labelled evaluation set: {labels['n_evaluated']} curated proteins",
    ]
    for evaluation in labels["evaluations"]:
        low, high = evaluation["accuracy_95ci"]
        lines.append(
            f"  {evaluation['name']:<18} accuracy {evaluation['accuracy']:.3f} "
            f"(95% CI {low:.3f}-{high:.3f})  macro-F1 {evaluation['macro_f1']:.3f}",
        )

    null = novelty["permutation_null"]
    lines.extend(
        [
            "",
            f"novel candidates: {novelty['candidates_after_quality_filter']} after quality control "
            f"({novelty['removed_by_quality_filter']} removed as incomplete)",
            f"  permutation null: {null['null_mean_candidates']} expected "
            f"(enrichment {null['enrichment_over_null']}x, p={null['p_value']}, "
            f"empirical FDR {null['empirical_fdr']})",
            "",
            f"architectures tested for taxonomic spread: {recurrence['n_architectures_tested']}",
            f"  significant at FDR<{recurrence['significance_fdr']}: {recurrence['n_significant_at_fdr']}",
        ],
    )
    return "\n".join(lines)
