"""Assemble the validation report: labels, quality, recurrence, and nulls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from domain_layout.constants import NOVELTY_THRESHOLD
from domain_layout.pipeline import LayoutRunConfig, analyze_store
from experimental.bakeoff import BakeoffInputs, run_bakeoff
from jdp_classifier.architecture import features_from_ida
from jdp_classifier.domain_evidence import ida_from_record
from jdp_classifier.rules import predict_class
from validation.capacity import capacity_report, sharp_decidability_report
from validation.chaperone_evidence import chaperone_report
from validation.clustering import clustering_report
from validation.grammar_signals import grammar_summary
from validation.homology import (
    LabelledProteins,
    baseline_notes,
    genus_blocked_predictions,
    profile_hmm_predictions,
)
from validation.labels import collect_gold_labels
from validation.metrics import evaluate, majority_baseline
from validation.nulls import novelty_null, score_threshold_sweep
from validation.ordering_stability import HEADLINE_ERROR_RATE, ResampleSettings, stability_report
from validation.quality import assess_records, summarize_quality
from validation.recurrence import benjamini_hochberg, genus_of, recurrence_with_null

if TYPE_CHECKING:
    from collections.abc import Mapping
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


def _n_ortholog_groups(layouts: list[ProteinLayout], labels: Mapping[str, str]) -> int:
    """Independent targets in the label set, counted as distinct protein names.

    The same subfamily member number in Homo, Mus, Bos and Rattus is one piece of
    evidence, not four: those orthologs are near-identical. Counting proteins instead
    overstates how much the benchmark can resolve - 122 of 129 labelled proteins sit in a
    cross-genus ortholog group.
    """
    names = {
        layout.record.name.strip() or layout.record.accession for layout in layouts if layout.record.accession in labels
    }
    return len(names) or len(labels)


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
    # The profile-HMM baseline is the one that matters: it is the standard method for this
    # task, so the layout classifier has to beat it to be worth its complexity. Predictions
    # are out-of-fold, since a profile built from the protein it later scores is not a
    # baseline at all.
    labelled_sequences = {
        layout.record.accession: layout.record.sequence
        for layout in complete_layouts
        if layout.record.accession in truth_complete
    }
    hmm_predictions = profile_hmm_predictions(
        truth_complete,
        labelled_sequences,
        seed=permutations.seed,
        msa_backend_name=run_config.msa_backend,
    )
    if hmm_predictions:
        baselines["profile_hmm_cv"] = hmm_predictions

    # The same baseline with every genus confined to one fold. A random split leaves close
    # homologs of a held-out protein in training, which helps a sequence-based profile far
    # more than an architecture-based rule system - so the random-split comparison is
    # biased towards the baseline, and the gap between these two numbers is the size of
    # that advantage.
    labelled_organisms = {
        layout.record.accession: layout.record.organism_name
        for layout in complete_layouts
        if layout.record.accession in truth_complete
    }
    blocked_predictions = genus_blocked_predictions(
        LabelledProteins(labels=truth_complete, sequences=labelled_sequences, organisms=labelled_organisms),
        seed=permutations.seed,
        msa_backend_name=run_config.msa_backend,
    )
    if blocked_predictions:
        baselines["profile_hmm_genus_blocked"] = blocked_predictions

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
        # How many of these methods the label set can actually place in a certified
        # order. Reported next to the rankings, because a ranking finer than the labels
        # can support is not a weak result but an undecidable one.
        "benchmark_capacity": capacity_report(
            len(truth_complete),
            evaluations,
            n_effective=_n_ortholog_groups(complete_layouts, truth_complete),
        ),
        # The same bound applied at its own sharpness rather than its crude form: a pair's
        # margin is charged only against targets the loser fails and the winner does not,
        # which is the hypothesis the impossibility construction actually requires.
        "sharp_decidability": sharp_decidability_report(
            truth_complete,
            {"domain_layout": predictions, **baselines},
            error_rate=HEADLINE_ERROR_RATE,
        ),
        # The same question asked without the theorem: flip labels at plausible error
        # rates and see whether the ranking survives. The bound is worst case and this is
        # average case, so the two are expected to differ; reporting both keeps the
        # distinction between "certifiable" and "probable" on the page rather than in a
        # reader's head.
        "ordering_stability": stability_report(
            truth_complete,
            {"domain_layout": predictions, **baselines},
            settings=ResampleSettings(seed=permutations.seed),
        ),
        # Redundancy control by sequence identity, on all three region definitions.
        # Genus blocking was the first attempt and it moved nothing, because most labelled
        # proteins sit in cross-genus ortholog groups - taxonomy is the wrong unit. This
        # reports how much redundancy actually survives at the conventional 30% threshold,
        # and how differently the three definitions answer that.
        # Whether the rarer architectures are chaperones at all. Past roughly the top 20
        # architectures a J-like domain need not carry the HPD motif that contacts Hsp70,
        # and a J-domain protein without it is doing something else - so an architecture
        # whose members mostly lack it should not be counted as a chaperone class.
        "chaperone_evidence": chaperone_report(complete_layouts),
        "redundancy_clustering": clustering_report(
            complete_layouts,
            restrict_to=sorted(truth_complete),
        ),
        "homology_baseline": {
            **baseline_notes(len(truth_complete), len(hmm_predictions)),
            "n_predicted_genus_blocked": len(blocked_predictions),
            "n_genera_in_labelled_set": len({genus_of(name) for name in labelled_organisms.values()}),
        },
        # Candidate models, measured against the incumbent on identical proteins. Nothing
        # here changes any number above it; a challenger is promoted only by the verdict.
        "bakeoff": run_bakeoff(
            complete_layouts,
            BakeoffInputs(
                labels=truth_complete,
                incumbent=predictions,
                homology=hmm_predictions,
                homology_genus_blocked=blocked_predictions,
                structural=run_config.structural_features,
            ),
            seed=permutations.seed,
        ),
        "sequence_grammar": grammar_summary(complete_layouts),
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


def _decidability_lines(report: dict[str, Any]) -> list[str]:
    """Render how much of the accuracy ranking is actually decidable.

    Printed rather than left in the JSON: a reader who sees only the accuracy table will
    over-read small gaps. Two numbers appear, and they are not redundant. The capacity
    bound is worst case - whether *any* arrangement of label errors within budget could
    reverse an ordering, which is what certification requires. The simulation beside it is
    average case - whether randomly drawn errors actually do. An ordering can be robust in
    practice and still uncertifiable, so both are stated.
    """
    capacity = report.get("benchmark_capacity", {})
    stability = report.get("ordering_stability", {})
    lines: list[str] = []
    if capacity:
        rate_key = f"{HEADLINE_ERROR_RATE:.0%}"
        headline = capacity["capacity_by_error_rate"].get(rate_key, {})
        lines.extend(
            [
                "",
                (
                    f"label-set capacity at {rate_key} annotation error: "
                    f"{headline.get('capacity_at_effective_n', '?')} methods can be placed in a "
                    f"certified order, and {capacity['n_methods_compared']} are compared "
                    f"(n={capacity['n_targets']}, effective n={capacity['n_effective_targets']})"
                ),
            ],
        )
        lines.extend(
            f"  undecidable: {pair['better']} over {pair['worse']} "
            f"(gap {pair['score_gap']}, needs label error below "
            f"{pair['max_error_rate_for_a_decidable_ordering']})"
            for pair in capacity["pairwise_decidability"][:3]
            if not pair["decidable_at"].get(rate_key, True)
        )
    sharp = report.get("sharp_decidability", {})
    if sharp:
        lines.append(
            f"  charged sharply (only targets the loser fails and the winner does not): "
            f"{sharp['n_decidable']} of {sharp['n_pairs']} orderings decidable at "
            f"{sharp['assumed_error_rate']:.0%}",
        )
    if stability:
        n_pairs = stability["n_methods"] * (stability["n_methods"] - 1) // 2
        lines.append(
            f"  simulated at the same {stability['headline_error_rate']:.0%} error: "
            f"{stability['n_unstable_at_headline_error']} of {n_pairs} orderings reverse. "
            f"The bound is worst case and the simulation is average case, so a pair can be "
            f"uncertifiable and still stable under randomly drawn errors.",
        )
    return lines


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

    grammar = report.get("sequence_grammar", {})
    homology = report.get("homology_baseline", {})
    bakeoff = report.get("bakeoff", {})

    if homology.get("n_predicted"):
        lines.append(
            f"  (profile-HMM baseline assigned {homology['n_predicted']} of "
            f"{homology['n_labelled']}; {homology['n_unassigned']} matched no profile)",
        )
    elif homology.get("backend") == "unavailable":
        lines.append("  (profile-HMM baseline skipped: pyhmmer not installed)")

    lines.extend(_decidability_lines(report))

    null = novelty["permutation_null"]
    lines.extend(
        [
            "",
            (
                f"novel candidates: {novelty['candidates_after_quality_filter']} after quality control "
                f"({novelty['removed_by_quality_filter']} removed as incomplete)"
            ),
            (
                f"  permutation null: {null['null_mean_candidates']} expected "
                f"(enrichment {null['enrichment_over_null']}x, p={null['p_value']}, "
                f"empirical FDR {null['empirical_fdr']})"
            ),
            "",
            f"architectures tested for taxonomic spread: {recurrence['n_architectures_tested']}",
            f"  significant at FDR<{recurrence['significance_fdr']}: {recurrence['n_significant_at_fdr']}",
        ],
    )

    if grammar.get("n_regions_profiled"):
        cross = grammar.get("novelty_cross_check", {})
        lines.extend(
            [
                "",
                (
                    f"sequence grammar: {grammar['n_regions_profiled']} region(s) profiled; "
                    f"{grammar['n_low_complexity']} repetitive, "
                    f"{grammar['n_compositionally_degenerate']} compositionally degenerate"
                ),
                (
                    f"  grammar-anomalous proteins (top {(1 - grammar['anomaly_percentile']) * 100:.0f}%): "
                    f"{grammar['n_grammar_anomalous_proteins']}"
                ),
                (
                    f"  overlap with novelty candidates: {cross.get('n_also_grammar_anomalous', 0)} observed vs "
                    f"{cross.get('expected_if_independent', 0)} expected if independent "
                    f"(enrichment {cross.get('enrichment', 0)}x)"
                ),
            ],
        )

    if bakeoff.get("status") == "evaluated":
        lines.extend(["", f"candidate-model bake-off (n={bakeoff['n_evaluated']}):"])
        for evaluation in bakeoff["evaluations"]:
            low, high = evaluation["accuracy_95ci"]
            lines.append(
                f"  {evaluation['name']:<20} accuracy {evaluation['accuracy']:.3f} "
                f"(95% CI {low:.3f}-{high:.3f})  macro-F1 {evaluation['macro_f1']:.3f}",
            )
        lines.append(f"  -> {bakeoff['verdict']['decision']}")
    elif bakeoff.get("status"):
        lines.extend(["", f"candidate-model bake-off: {bakeoff['status']}"])

    return "\n".join(lines)
