"""CLI for the sequence-structure-function coupling analysis (``analyze-coupling``).

The question is which of the three has moved. Structure is far more conserved than
sequence, so pairs fall into regimes: sequence diverged while the fold held, both held,
or - rarest and most interesting - the fold moved while the sequence did not. The regime a
pair sits in says something the pairwise identity alone does not.

Structure comes from Foldseek rather than Dali. Dali is the more established method and is
what a reviewer will ask about, but it is far too slow for an all-versus-all over 142,948
AlphaFold models; Foldseek reaches the same fold-level conclusions through a structural
alphabet at a speed that makes the comparison possible at all.

Functional terms are optional. When supplied they mark whether a pair shares any annotation
- a coarse test, since two chaperones both annotated "protein folding" share a term without
sharing a role, so it is reported beside the regime rather than used to define it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from domain_layout.records import load_domain_store
from structure_analysis.structural_search import (
    SearchOptions,
    foldseek_available,
    readable_structure_count,
    search_structures,
    stage_structures,
)
from validation.coupling import classify_pairs, coupling_report
from validation.representatives import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    SelectionOptions,
    choose_representatives,
    representative_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def load_function_terms(path: Path | None) -> dict[str, list[str]] | None:
    """Read GO terms per accession from a function store, if one is available.

    Returns ``None`` rather than an empty mapping when no store is given, because the two
    mean different things downstream: ``None`` suppresses the shared-function column
    entirely, while an empty mapping would report every pair as not sharing function.
    """
    if path is None or not path.exists():
        logger.warning("no function store supplied; the shared-function column will be omitted")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms: dict[str, list[str]] = {}
    for accession, record in payload.get("records", {}).items():
        collected: list[str] = []
        go_terms = record.get("go_terms", {})
        if isinstance(go_terms, dict):
            for aspect_terms in go_terms.values():
                collected.extend(str(item) for item in aspect_terms or ())
        elif isinstance(go_terms, list):
            collected.extend(str(item) for item in go_terms)
        if collected:
            terms[accession] = sorted(set(collected))
    logger.info("loaded functional terms for %s proteins from %s", len(terms), path)
    return terms


def _mean_plddt(path: Path | None) -> dict[str, float]:
    """Per-accession mean pLDDT, so the best-resolved member represents each group."""
    if path is None or not path.exists():
        return {}
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            accession = (row.get("accession") or "").strip()
            raw = row.get("mean_plddt")
            if not accession or raw in (None, ""):
                continue
            try:
                scores[accession] = float(raw)
            except ValueError:
                continue
    logger.info("loaded mean pLDDT for %s proteins from %s", len(scores), path)
    return scores


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze-coupling`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify structural pairs by how sequence and structure diverged, and report where function tracks them."
        ),
    )
    parser.add_argument("structures", type=Path, help="Directory of AlphaFold models (PDB or CIF)")
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="Directory to search against (default: an all-versus-all within --structures)",
    )
    parser.add_argument("--functions", type=Path, default=None, help="function_store.json for shared-term marking")
    parser.add_argument("-o", "--output", type=Path, default=Path("coupling_report.json"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--foldseek", default=None, help="Path to the foldseek binary")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help=(
            "Where to decompress gzipped models. AlphaFold ships .pdb.gz / .cif.gz, which "
            "Foldseek's directory scan skips without error - staging is required, not "
            "optional, when the source holds compressed files."
        ),
    )
    parser.add_argument("--stage-suffix", default=".pdb", choices=[".pdb", ".cif"])
    parser.add_argument(
        "--domain-store",
        type=Path,
        default=None,
        help="Domain store, required to select representatives rather than staging everything",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=list(STRATEGIES),
        help=(
            "How to pick which proteins enter the comparison. 'architecture' is one per "
            "domain architecture - this project's own unit; 'species'/'genus' remove "
            "phylogenetic over-representation; 'random' ignores both, and agreement with "
            "it is evidence a result is not an artefact of the choice."
        ),
    )
    parser.add_argument("--structure-features", type=Path, default=None, help="CSV supplying mean pLDDT")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-structures", type=int, default=None, help="Cap models staged (for a trial run)")
    parser.add_argument(
        "--max-staged-bytes",
        type=int,
        default=60 * 10**9,
        help=(
            "Ceiling on decompressed bytes. The default suits node-local scratch; lower it "
            "sharply when staging onto a shared filesystem under a group quota."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the coupling analysis and write the report."""
    args = build_parser().parse_args(argv)

    options = SearchOptions(threads=max(1, args.threads))
    if args.foldseek:
        options = SearchOptions(executable=args.foldseek, threads=max(1, args.threads))

    if not foldseek_available(options.executable):
        sys.stderr.write(
            f"foldseek not found at '{options.executable}'.\n"
            "Install it under vendor/foldseek or pass --foldseek; the coupling analysis "
            "cannot run without a structural search.\n",
        )
        raise SystemExit(2)

    # Which proteins enter the comparison at all. An all-versus-all over every model is
    # ~10^10 pairs and, staged, exceeds the shared quota; it would also weight the result
    # toward whichever families happen to be over-sequenced.
    wanted: list[str] | None = None
    selection: dict[str, object] = {}
    if args.domain_store is not None:
        store = load_domain_store(args.domain_store)
        quality = _mean_plddt(args.structure_features)
        wanted = choose_representatives(
            store,
            options=SelectionOptions(
                strategy=args.strategy,
                limit=args.max_structures,
                seed=args.selection_seed,
            ),
            quality=quality,
            restrict_to=sorted(quality) if quality else None,
        )
        selection = representative_summary(store, wanted, strategy=args.strategy)
        logger.info("%s representatives under the %s strategy", len(wanted), args.strategy)

    structures = args.structures
    if readable_structure_count(structures) == 0 and args.stage_dir is not None:
        logger.info("no readable structures in %s; staging decompressed copies", structures)
        staged = stage_structures(
            structures,
            args.stage_dir,
            suffix=args.stage_suffix,
            limit=args.max_structures,
            accessions=wanted,
            max_bytes=args.max_staged_bytes,
        )
        if staged == 0:
            sys.stderr.write(f"nothing staged from {structures}; check the source directory.\n")
            raise SystemExit(1)
        structures = args.stage_dir

    targets = args.targets if args.targets is not None else structures
    logger.info("searching %s against %s", structures, targets)
    hits = search_structures(structures, targets, options=options)
    logger.info("%s structural hits above threshold", len(hits))

    pairs = classify_pairs(hits, function_terms=load_function_terms(args.functions))
    report = coupling_report(pairs, max_examples=args.max_examples)
    report["inputs"] = {
        "structures": str(structures),
        "targets": str(targets),
        "n_hits": len(hits),
        "n_pairs_classified": len(pairs),
        "functions_supplied": args.functions is not None and args.functions.exists(),
        "selection": selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stderr.write(f"\nWrote {args.output}\n")


if __name__ == "__main__":
    main()
