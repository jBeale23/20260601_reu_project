"""CLI for AlphaFold structural features (``analyze-protein-structures``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from domain_layout.records import load_domain_store
from structure_analysis.batch_features import compute_all, write_structure_features


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze-protein-structures`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute AlphaFold structural features (pLDDT bands, compactness, contact "
            "order, burial, surface versus buried charge) for every protein in a domain store."
        ),
    )
    parser.add_argument("domain_store", type=Path, help="Domain store JSON from fetch-protein-domains")
    parser.add_argument("--structures", type=Path, required=True, help="Directory of AlphaFold model files")
    parser.add_argument("-o", "--output", type=Path, default=Path("structure_features.csv"))
    parser.add_argument("--workers", type=int, default=1, help="Processes to use (default: 1)")
    parser.add_argument("--max-proteins", type=int, default=None, help="Cap for a smoke test")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for analyze-protein-structures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if not args.domain_store.is_file():
        parser.error(f"Domain store not found: {args.domain_store}")
    if not args.structures.is_dir():
        parser.error(f"Structures directory not found: {args.structures}")

    store = load_domain_store(args.domain_store)
    accessions = sorted(store.proteins)
    if args.max_proteins is not None:
        accessions = accessions[: args.max_proteins]

    rows, summary = compute_all(accessions, args.structures, workers=max(1, args.workers))
    write_structure_features(rows, args.output)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.to_json_dict(), indent=2) + "\n", encoding="utf-8")

    sys.stderr.write(
        f"Structural features for {summary.n_with_model} of {summary.n_requested} protein(s) "
        f"({summary.coverage:.1%} coverage); {summary.n_missing} had no model.\n"
        f"Wrote {args.output} and {summary_path}\n",
    )


if __name__ == "__main__":
    main()
