"""CLI for the domain-layout dual-layer pipeline (``analyze-domain-layout``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from domain_layout import disorder as disorder_backend
from domain_layout import msa as msa_module
from domain_layout import shark as shark_backend
from domain_layout.pipeline import (
    LayoutRunConfig,
    analyze_store,
    load_reference_regions,
    write_outputs,
)
from domain_layout.records import load_domain_store
from domain_layout.reference_data import default_reference_classes, default_reference_store

if TYPE_CHECKING:
    from domain_layout.profiles import ReferenceRegion

_DEFAULT_OUTPUT_DIR = Path("domain_layout_results")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze-domain-layout`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Split every protein into structured (MSA) and unalignable (SHARK) regions, "
            "then classify it as JDP class A/B/C, a subclass, or a novel-category candidate."
        ),
    )
    parser.add_argument(
        "domain_store",
        type=Path,
        nargs="?",
        default=None,
        help="Domain store JSON written by fetch-protein-domains",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for region/feature CSVs, summary JSON, and subFASTAs (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--disorder-backend",
        choices=(
            disorder_backend.BACKEND_AUTO,
            disorder_backend.BACKEND_METAPREDICT,
            disorder_backend.BACKEND_FOLDINDEX,
        ),
        default=disorder_backend.BACKEND_AUTO,
        help="Disorder predictor: auto uses metapredict when installed, else FoldIndex",
    )
    parser.add_argument(
        "--shark-backend",
        choices=(shark_backend.BACKEND_AUTO, shark_backend.BACKEND_BIO_SHARK, shark_backend.BACKEND_KMER),
        default=shark_backend.BACKEND_AUTO,
        help="Alignment-free scorer: auto uses bio-shark when installed, else BLOSUM k-mer scores",
    )
    parser.add_argument(
        "--reference-store",
        type=Path,
        default=None,
        help="Curated reference domain store (default: bundled data/reference_jdps)",
    )
    parser.add_argument(
        "--reference-classes",
        type=Path,
        default=None,
        help="JSON map of reference accession to class (default: bundled data/reference_jdps)",
    )
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="Skip SHARK comparison against reference JDPs",
    )
    parser.add_argument("--no-fasta", action="store_true", help="Do not write subFASTA files")
    parser.add_argument(
        "--align-msa",
        action="store_true",
        help="Align the MSA-routed subFASTAs with MAFFT and write them to subfastas/msa_aligned",
    )
    parser.add_argument(
        "--msa-backend",
        choices=(msa_module.BACKEND_AUTO, msa_module.BACKEND_MAFFT, msa_module.BACKEND_PROGRESSIVE),
        default=msa_module.BACKEND_AUTO,
        help="Aligner for --align-msa (default: mafft when installed)",
    )
    parser.add_argument(
        "--msa-threads",
        type=int,
        default=1,
        help="Threads passed to MAFFT (default: 1, which keeps runs byte-reproducible)",
    )
    parser.add_argument(
        "--max-aligned-per-family",
        type=int,
        default=2000,
        help="Cap on sequences aligned per family; larger families are randomly sub-sampled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for per-family sub-sampling when --align-msa is set",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Analyze proteins across N processes (default: 1). Results are unchanged.",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Analyze only the first N proteins (smoke tests)",
    )
    parser.add_argument(
        "--show-backends",
        action="store_true",
        help="Print which disorder/SHARK backends are active and exit",
    )
    return parser


def _resolve_reference_paths(args: argparse.Namespace) -> tuple[Path, Path] | None:
    if args.no_references:
        return None
    store = args.reference_store or default_reference_store()
    classes = args.reference_classes or default_reference_classes()
    if not store.is_file() or not classes.is_file():
        return None
    return store, classes


def main(argv: list[str] | None = None) -> None:
    """Console entry point for analyze-domain-layout."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Progress goes to stderr so a long cluster run shows life in its log file.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if args.show_backends:
        sys.stderr.write(
            f"disorder_backend={disorder_backend.active_backend(backend=args.disorder_backend)} "
            f"(metapredict_installed={disorder_backend.metapredict_available()})\n"
            f"shark_backend={shark_backend.active_backend(backend=args.shark_backend)} "
            f"(bio_shark_installed={shark_backend.shark_available()})\n",
        )
        return

    if args.domain_store is None:
        parser.error("domain_store is required (or use --show-backends)")
    if not args.domain_store.is_file():
        parser.error(f"Domain store not found: {args.domain_store}")

    try:
        store = load_domain_store(args.domain_store)
    except ValueError as exc:
        parser.error(str(exc))

    if len(store) == 0:
        sys.stderr.write("WARNING: domain store contains no protein records.\n")

    references: list[ReferenceRegion] = []
    reference_paths = _resolve_reference_paths(args)
    if reference_paths is None:
        if not args.no_references:
            sys.stderr.write(
                "WARNING: no reference JDP profiles found; SHARK similarity columns will be empty.\n",
            )
    else:
        try:
            references = load_reference_regions(
                reference_paths[0],
                reference_paths[1],
                backend=args.disorder_backend,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

    config = LayoutRunConfig(
        disorder_backend=args.disorder_backend,
        shark_backend=args.shark_backend,
        write_fastas=not args.no_fasta,
        align_msa=args.align_msa,
        msa_backend=args.msa_backend,
        msa_threads=max(1, args.msa_threads),
        max_aligned_per_family=args.max_aligned_per_family,
        seed=args.seed,
        max_proteins=args.max_proteins,
        workers=max(1, args.workers),
    )
    result = analyze_store(store, config=config, references=references)
    summary = write_outputs(result, args.output_dir, write_fastas=config.write_fastas, config=config)

    sys.stderr.write(
        f"Analyzed {summary['n_proteins']} protein(s) "
        f"(disorder={summary['disorder_backend']}, shark={summary['shark_backend']}, "
        f"{summary['n_reference_regions']} reference region(s)).\n",
    )
    sys.stderr.write(
        f"Classes: {summary['class_counts']}; novel candidates: {summary['n_novel_candidates']}.\n",
    )
    if summary["n_skipped_without_sequence"]:
        sys.stderr.write(
            f"WARNING: {summary['n_skipped_without_sequence']} record(s) had no sequence and were skipped.\n",
        )
    sys.stderr.write(f"Wrote outputs to {args.output_dir}\n")


if __name__ == "__main__":
    main()
