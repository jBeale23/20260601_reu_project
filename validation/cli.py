"""CLI for the validation suite (``validate-jdp-classification``)."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from domain_layout import disorder as disorder_backend
from domain_layout import shark as shark_backend
from domain_layout.pipeline import LayoutRunConfig, load_reference_regions
from domain_layout.records import load_domain_store
from domain_layout.reference_data import default_reference_classes, default_reference_store
from validation.report import PermutationSettings, build_validation_report, format_summary, write_report

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path("validation_report.json")


def load_structural_features(path: Path | None) -> dict[str, dict[str, float]]:
    """Read per-protein structural features from the CSV the structure stage writes.

    Returns an empty mapping when no path is given, which is what leaves the bake-off
    grammar-only. That default is why the structural challengers had never run: the
    features were computed and written, but nothing on the command line could hand them
    to the report, so ``structural_features`` was always empty.

    Non-numeric columns are skipped rather than coerced - ``accession`` keys the row, and
    a flag like ``is_largely_disordered`` arrives as a string that would otherwise become
    a silent zero.
    """
    if path is None:
        return {}
    features: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            values: dict[str, float] = {}
            for column, raw in row.items():
                if column == "accession" or raw is None or raw == "":
                    continue
                try:
                    values[column] = float(raw)
                except ValueError:
                    continue
            if values:
                features[accession] = values
    logger.info("loaded structural features for %s proteins from %s", len(features), path)
    return features


def build_parser() -> argparse.ArgumentParser:
    """Build the ``validate-jdp-classification`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate JDP classification: quality control, calibration against curated "
            "labels, permutation nulls for the novelty call, and cross-species recurrence."
        ),
    )
    parser.add_argument("domain_store", type=Path, help="Domain store JSON from fetch-protein-domains")
    parser.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUTPUT, help="Report JSON path")
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="Permutations for the novelty null (default: 200)",
    )
    parser.add_argument(
        "--recurrence-permutations",
        type=int,
        default=1000,
        help="Permutations for the taxonomic-spread null (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the permutation tests")
    parser.add_argument(
        "--disorder-backend",
        choices=(
            disorder_backend.BACKEND_AUTO,
            disorder_backend.BACKEND_METAPREDICT,
            disorder_backend.BACKEND_FOLDINDEX,
        ),
        default=disorder_backend.BACKEND_AUTO,
    )
    parser.add_argument(
        "--shark-backend",
        choices=(shark_backend.BACKEND_AUTO, shark_backend.BACKEND_BIO_SHARK, shark_backend.BACKEND_KMER),
        default=shark_backend.BACKEND_AUTO,
    )
    parser.add_argument(
        "--structural-features",
        type=Path,
        default=None,
        help=(
            "CSV of per-protein AlphaFold structural features, as written by "
            "analyze-protein-structures. Supplying it enables the two structural "
            "challengers; without it the bake-off is grammar-only."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Processes for the layout stage")
    parser.add_argument("--no-references", action="store_true", help="Skip SHARK reference comparison")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for validate-jdp-classification."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if not args.domain_store.is_file():
        parser.error(f"Domain store not found: {args.domain_store}")

    try:
        store = load_domain_store(args.domain_store)
    except ValueError as exc:
        parser.error(str(exc))

    references = []
    if not args.no_references:
        store_path, classes_path = default_reference_store(), default_reference_classes()
        if store_path.is_file() and classes_path.is_file():
            references = load_reference_regions(store_path, classes_path, backend=args.disorder_backend)
        else:
            sys.stderr.write("WARNING: bundled reference JDPs not found; SHARK novelty term will not fire.\n")

    report = build_validation_report(
        store,
        references=references,
        config=LayoutRunConfig(
            disorder_backend=args.disorder_backend,
            shark_backend=args.shark_backend,
            write_fastas=False,
            workers=max(1, args.workers),
            structural_features=load_structural_features(args.structural_features),
        ),
        permutations=PermutationSettings(
            novelty=args.permutations,
            recurrence=args.recurrence_permutations,
            seed=args.seed,
        ),
    )
    write_report(report, args.output)

    sys.stderr.write(format_summary(report) + "\n")
    sys.stderr.write(f"\nWrote {args.output}\n")


if __name__ == "__main__":
    main()
