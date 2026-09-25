"""CLI for DnaJ conserved charge / motif window analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from domain_layout import msa as msa_backend
from domain_layout.records import load_domain_store
from motif_conservation.analyze import MotifRunConfig, analyze_dnaj_fetch, write_outputs
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
        "--domain-json",
        type=Path,
        default=None,
        help=(
            "Domain store from fetch-protein-domains. Required for real fetch JSON: the "
            "architecture fetch carries no sequences or domain coordinates to slice."
        ),
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
        help="Cap on sequences aligned per family; larger families are randomly sub-sampled",
    )
    parser.add_argument(
        "--msa-backend",
        choices=(msa_backend.BACKEND_AUTO, msa_backend.BACKEND_MAFFT, msa_backend.BACKEND_PROGRESSIVE),
        default=msa_backend.BACKEND_AUTO,
        help="Aligner for structured domain families (default: mafft when installed)",
    )
    parser.add_argument(
        "--msa-threads",
        type=int,
        default=1,
        help="Threads passed to MAFFT (default: 1, which keeps runs byte-reproducible)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for per-family sub-sampling, so a capped run is reproducible",
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

    if args.fetch_json is None and args.domain_json is None:
        parser.error("Provide a DnaJ fetch JSON, --domain-json, or both.")

    if args.fetch_json is not None and not args.fetch_json.is_file():
        msg = f"Fetch JSON not found: {args.fetch_json}"
        raise SystemExit(msg)

    if args.window_min < 1 or args.window_max < args.window_min:
        msg = "Invalid window range"
        raise SystemExit(msg)

    domain_store = None
    if args.domain_json is not None:
        if not args.domain_json.is_file():
            msg = f"Domain store not found: {args.domain_json}"
            raise SystemExit(msg)
        domain_store = load_domain_store(args.domain_json)
        sys.stderr.write(f"Loaded {len(domain_store)} domain record(s) from {args.domain_json}\n")

    window_lengths = tuple(range(args.window_min, args.window_max + 1))
    summaries, accession_rows, curves = analyze_dnaj_fetch(
        args.fetch_json,
        domain_store=domain_store,
        config=MotifRunConfig(
            window_lengths=window_lengths,
            min_members=args.min_members,
            max_per_family=args.max_per_family,
            msa_backend=args.msa_backend,
            msa_threads=max(1, args.msa_threads),
            seed=args.seed,
        ),
    )
    if not accession_rows:
        sys.stderr.write(
            "WARNING: no domain slices found. Architecture fetch JSON has no sequences; "
            "pass --domain-json from fetch-protein-domains.\n",
        )
    write_outputs(args.output_dir, summaries, accession_rows, curves)
    resolved_msa = msa_backend.active_backend(backend=args.msa_backend)
    if resolved_msa != msa_backend.BACKEND_MAFFT and args.msa_backend != msa_backend.BACKEND_PROGRESSIVE:
        sys.stderr.write(
            "WARNING: mafft not found on PATH; structured families were aligned with the "
            "progressive fallback, which is weaker. Install MAFFT or load its module.\n",
        )
    sys.stderr.write(
        f"Wrote motif outputs for {len(summaries)} family(ies) "
        f"({len(accession_rows)} accession rows) to {args.output_dir}; MSA backend: {resolved_msa}\n",
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
