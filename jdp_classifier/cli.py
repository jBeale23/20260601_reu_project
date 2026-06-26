"""CLI for rule-based JDP classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jdp_classifier.classify import classify_fetch_json, write_classifications_csv
from scripts.extract_uniprot_ids import fetch_warnings


def main() -> None:
    """Console entry point for classify-jdp."""
    parser = argparse.ArgumentParser(
        description="Rule-based JDP class A/B/C prediction with HPD, localization, and layout tags.",
    )
    parser.add_argument("fetch_json", type=Path, help="DnaJ fetch output JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output classification CSV path",
    )
    parser.add_argument(
        "--dnaj-rows",
        choices=("dedupe", "explode"),
        default="dedupe",
        help="One row per accession (dedupe) or per architecture membership (explode)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip UniProt FASTA fallback when InterPro sequence is missing",
    )
    parser.add_argument(
        "--min-confidence",
        choices=("high", "medium", "low"),
        default=None,
        help="Only include rows at or above this class confidence tier",
    )
    args = parser.parse_args()

    if not args.fetch_json.is_file():
        parser.error(f"Fetch JSON not found: {args.fetch_json}")

    try:
        data = json.loads(args.fetch_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON in {args.fetch_json}: {exc.msg}")

    if not data.get("architectures"):
        parser.error("Input JSON does not look like DnaJ architecture fetch output.")

    for warning in fetch_warnings(data):
        sys.stderr.write(f"WARNING: {warning}\n")

    results, uniprot_fetches = classify_fetch_json(
        data,
        dnaj_rows=args.dnaj_rows,
        allow_fetch=not args.no_fetch,
        min_confidence=args.min_confidence,
    )

    if not results:
        sys.stderr.write("WARNING: No classification rows produced.\n")

    write_classifications_csv(results, args.output)

    sys.stderr.write(
        f"Classified {len(results)} row(s); UniProt FASTA fallback used {uniprot_fetches} time(s).\n",
    )
    sys.stderr.write(f"Wrote {args.output}\n")


if __name__ == "__main__":
    main()
