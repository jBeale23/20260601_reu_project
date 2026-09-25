"""CLI for the neural challenger (``train-neural-challenger``).

Deliberately not wired into :mod:`validation.report`. The report is the incumbent record and
runs on CPU in minutes; this needs a GPU and tens of minutes per seed, and folding it in
would make every validation run depend on hardware most of them do not have. It writes its
own report, which the bake-off can pick up when a run exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from domain_layout.pipeline import LayoutRunConfig, analyze_store
from domain_layout.records import load_domain_store
from experimental.neural.architecture import TrainingConfig, torch_available
from experimental.neural.data import build_examples
from experimental.neural.train import neural_report
from validation.clustering import (
    DEFAULT_IDENTITY_THRESHOLD,
    REGION_NON_J,
    cluster_blocked_folds,
    cluster_layouts,
)
from validation.labels import collect_gold_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Below two clusters there is no held-out set: one cluster means every fold holds
# every protein, leaving nothing to train on.
MIN_CLUSTERS_FOR_EVALUATION = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the ``train-neural-challenger`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the hierarchical neural challenger - grammar CNN/BiLSTM, syntax "
            "transformer, dual heads - over identity-clustered folds."
        ),
    )
    parser.add_argument("domain_store", type=Path, help="Domain store JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("neural_report.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cluster-region",
        default=REGION_NON_J,
        help=(
            "Region used to define redundancy. Defaults to the non-J-domain regions, which "
            "is the partition matching what the classifier claims to read."
        ),
    )
    parser.add_argument("--identity-threshold", type=float, default=DEFAULT_IDENTITY_THRESHOLD)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Train the challenger and write its report."""
    args = build_parser().parse_args(argv)

    if not torch_available():
        sys.stderr.write("torch is not installed; the neural challenger cannot run.\n")
        raise SystemExit(2)

    store = load_domain_store(args.domain_store)
    # collect_gold_labels returns GoldLabel records carrying the evidence they came from;
    # the model needs the bare class string. Passing the record through instead made the
    # label space unsortable and failed the run after the whole layout stage had completed.
    gold = collect_gold_labels(store)
    truth = {accession: record.label for accession, record in gold.items()}
    if not truth:
        sys.stderr.write("no gold labels in this store; nothing to train against.\n")
        raise SystemExit(1)
    logger.info("%s labelled proteins", len(truth))

    result = analyze_store(store, config=LayoutRunConfig(write_fastas=False, workers=max(1, args.workers)))
    layouts = [layout for layout in result.layouts if layout.record.accession in truth]

    examples, vocabulary, feature_names = build_examples(layouts, truth)
    logger.info("%s examples, %s syntax tokens, %s static features", len(examples), len(vocabulary), len(feature_names))
    if not examples:
        sys.stderr.write("no usable examples after profile construction.\n")
        raise SystemExit(1)

    # Folds are clustered by sequence identity, not random. A model with this many
    # parameters memorises, and random folds leave near-identical relatives of every
    # held-out protein in training, which a memorising model scores well on.
    assignment = cluster_layouts(
        layouts,
        region=args.cluster_region,
        threshold=args.identity_threshold,
        restrict_to=[item.accession for item in examples],
    )
    folds = cluster_blocked_folds(
        assignment.representative_of,
        [item.accession for item in examples],
        n_folds=args.folds,
    )
    logger.info(
        "%s clusters over %s proteins at %.0f%% identity (%s)",
        assignment.n_clusters,
        assignment.n_sequences,
        args.identity_threshold * 100,
        args.cluster_region,
    )

    if assignment.n_clusters < MIN_CLUSTERS_FOR_EVALUATION:
        sys.stderr.write(
            f"identity clustering produced {assignment.n_clusters} cluster over "
            f"{assignment.n_sequences} proteins at {args.identity_threshold:.0%} on "
            f"{args.cluster_region}: every protein is redundant with every other, so no "
            f"held-out evaluation exists. Raise --identity-threshold or pick a different "
            f"--cluster-region.\n",
        )
        raise SystemExit(3)

    report = neural_report(
        examples,
        folds,
        n_syntax_tokens=len(vocabulary) + 2,
        settings=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seeds=tuple(args.seeds),
            device=args.device,
        ),
    )
    report["clustering"] = assignment.to_json_dict()
    report["n_static_features"] = len(feature_names)
    report["n_syntax_vocabulary"] = len(vocabulary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stderr.write(
        f"\n{report['model']}: mean accuracy {report['accuracy_mean']} "
        f"(spread {report['accuracy_spread']} over {report['n_seeds']} seeds)\n"
        f"{report['trainable_parameters']:,} trainable parameters for {report['n_examples']} examples "
        f"({report['parameters_per_labelled_example']} per example)\n"
        f"Wrote {args.output}\n",
    )


if __name__ == "__main__":
    main()
