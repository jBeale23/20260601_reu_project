"""CLI for DnaJ conserved charge / motif window analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from motif_conservation.analyze import analyze_dnaj_fetch, write_outputs
from motif_conservation.constants import DEFAULT_WINDOW_LENGTHS, MIN_FAMILY_MEMBERS
from motif_conservation.transfer import held_out_overlap_report


def build_parser() -> argparse.ArgumentParser:
    """Build the analyze-motif-conservation argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Align DnaJ domain families, sweep charge windows for cross-homolog conservation, "
            "and segment IDR/G/F-like block grammars."
        ),
    )
    parser.add_argument(
        "fetch_json",
        type=Path,
        nargs="?",
        default=None,
        help="DnaJ InterPro architecture fetch JSON",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("motif_results"),
        help="Directory for summary CSV, accession features, and curves JSON",
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=MIN_FAMILY_MEMBERS,
        help="Minimum unique accessions required per domain family",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=None,
        help="Optional cap on sequences per family (useful for Rockfish smoke tests)",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=min(DEFAULT_WINDOW_LENGTHS),
    )
    parser.add_argument(
        "--window-max",
        type=int,
        default=max(DEFAULT_WINDOW_LENGTHS),
    )
    parser.add_argument(
        "--held-out-pocket-csv",
        type=Path,
        default=None,
        help="Optional pocket_charge_summary.csv for held-out charge-inversion overlap report",
    )
    parser.add_argument(
        "--held-out-domain-family",
        default=None,
        help="Optional domain_family filter for held-out overlap",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console entry point for analyze-motif-conservation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fetch_json is None:
        parser.error("fetch_json is required unless only documenting usage")

    if not args.fetch_json.is_file():
        msg = f"Fetch JSON not found: {args.fetch_json}"
        raise SystemExit(msg)

    if args.window_min < 1 or args.window_max < args.window_min:
        msg = "Invalid window range"
        raise SystemExit(msg)

    window_lengths = tuple(range(args.window_min, args.window_max + 1))
    summaries, accession_rows, curves = analyze_dnaj_fetch(
        args.fetch_json,
        window_lengths=window_lengths,
        min_members=args.min_members,
        max_per_family=args.max_per_family,
    )
    write_outputs(args.output_dir, summaries, accession_rows, curves)
    sys.stderr.write(
        f"Wrote motif outputs for {len(summaries)} family(ies) "
        f"({len(accession_rows)} accession rows) to {args.output_dir}\n",
    )

    if args.held_out_pocket_csv is not None:
        report = held_out_overlap_report(
            args.output_dir / "motif_accession_features.csv",
            args.held_out_pocket_csv,
            domain_family=args.held_out_domain_family,
        )
        report_path = args.output_dir / "held_out_charge_inversion_overlap.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        sys.stderr.write(
            f"Held-out overlap: {report['n_overlap']} / "
            f"{report['n_charge_inversion_candidates']} inversion candidates "
            f"(wrote {report_path})\n",
        )


if __name__ == "__main__":
    main()
