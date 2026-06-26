"""CLI for binding-pocket charge analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from structure_analysis.analyze import (
    BatchAnalysisOptions,
    PocketChargeResult,
    _result_to_dict,
    analyze_directory,
    analyze_structure_file,
    export_contact_attribution_csv,
    export_pocket_residues,
    merge_results_directory,
    run_sensitivity_analysis,
    write_result,
)
from structure_analysis.pocket_reference import load_pocket_reference
from structure_analysis.quality import ConfidenceTier, tier_rank


def _default_pocket_ref() -> Path:
    return Path("data/pocket_refs/dnak_sbd_pocket.yaml")


def _default_reference_structure(pocket_ref_path: Path) -> Path:
    pocket_ref = load_pocket_reference(pocket_ref_path)
    return Path(f"data/dev_structures/AF-{pocket_ref.reference_accession}-F1-model_v6.pdb")


def _print_result_summary(result: object) -> None:
    if not isinstance(result, PocketChargeResult):
        return
    sys.stderr.write(
        f"  {result.accession} [mapping={result.mapping_confidence}/conservation={result.conservation_score}]: "
        f"contact={result.contact_net_charge} (Δ{result.delta_contact_net_charge}) "
        f"shell={result.shell_net_charge} (Δ{result.delta_net_charge_vs_reference}) "
        f"mean_ca={result.mean_contact_ca_distance}Å "
        f"id={result.contact_sequence_identity:.0%}\n",
    )


def _handle_merge_results(
    args: argparse.Namespace,
    *,
    min_confidence: ConfidenceTier | None,
    min_mapping_confidence: ConfidenceTier | None,
) -> None:
    if not args.merge_results.is_dir():
        msg = f"Merge results directory not found: {args.merge_results}"
        raise SystemExit(msg)
    results = merge_results_directory(
        args.merge_results,
        args.output_dir,
        min_confidence=min_confidence,
        min_mapping_confidence=min_mapping_confidence,
    )
    sys.stderr.write(
        f"Merged {len(results)} result(s) from {args.merge_results}; summary written to {args.output_dir}\n",
    )
    sys.stderr.write(f"  CSV: {args.output_dir / 'pocket_charge_summary.csv'}\n")


def _handle_sensitivity(args: argparse.Namespace, ref_structure: Path) -> None:
    if not args.structure.is_dir():
        msg = "--sensitivity requires a directory path"
        raise SystemExit(msg)
    sensitivity_dir = args.output_dir / "sensitivity"
    report_path = run_sensitivity_analysis(
        args.structure,
        args.reference_pocket,
        ref_structure,
        sensitivity_dir,
    )
    sys.stderr.write(f"Wrote sensitivity report to {report_path}\n")
    sys.stderr.write(f"  Stability: {sensitivity_dir / 'sensitivity_stability.csv'}\n")


def _handle_batch(
    args: argparse.Namespace,
    ref_structure: Path,
    *,
    min_confidence: ConfidenceTier | None,
    min_mapping_confidence: ConfidenceTier | None,
) -> None:
    if not args.structure.is_dir():
        msg = "--batch requires a directory path"
        raise SystemExit(msg)
    results = analyze_directory(
        BatchAnalysisOptions(
            structures_dir=args.structure,
            pocket_ref_path=args.reference_pocket,
            reference_structure_path=ref_structure,
            output_dir=args.output_dir,
            min_confidence=min_confidence,
            min_mapping_confidence=min_mapping_confidence,
        ),
    )
    sys.stderr.write(f"Analyzed {len(results)} structure(s); wrote results to {args.output_dir}\n")
    sys.stderr.write(f"  CSV: {args.output_dir / 'pocket_charge_summary.csv'}\n")
    for result in results:
        _print_result_summary(result)


def _handle_single_structure(
    args: argparse.Namespace,
    ref_structure: Path,
    *,
    min_confidence: ConfidenceTier | None,
    min_mapping_confidence: ConfidenceTier | None,
) -> None:
    if not args.structure.is_file():
        msg = f"Structure file not found: {args.structure}"
        raise SystemExit(msg)

    result = analyze_structure_file(args.structure, args.reference_pocket, ref_structure)

    if args.export_pocket_residues is not None:
        export_pocket_residues(
            args.structure,
            args.reference_pocket,
            ref_structure,
            args.export_pocket_residues,
        )
        sys.stderr.write(f"Wrote pocket residue export to {args.export_pocket_residues}\n")

    if args.export_contact_attribution is not None:
        export_contact_attribution_csv(result, args.export_contact_attribution)
        sys.stderr.write(f"Wrote contact attribution to {args.export_contact_attribution}\n")

    if min_confidence is not None and tier_rank(result.confidence_tier) < tier_rank(min_confidence):
        sys.stderr.write(
            f"WARNING: Result confidence tier '{result.confidence_tier}' is below --min-confidence {min_confidence}\n",
        )

    if min_mapping_confidence is not None and tier_rank(result.mapping_confidence) < tier_rank(min_mapping_confidence):
        sys.stderr.write(
            f"WARNING: Result mapping confidence '{result.mapping_confidence}' is below "
            f"--min-mapping-confidence {min_mapping_confidence}\n",
        )

    if args.output is None:
        sys.stdout.write(json.dumps(_result_to_dict(result), indent=2) + "\n")
    else:
        write_result(result, args.output)
        sys.stderr.write(f"Wrote {args.output}\n")

    for warning in result.warnings:
        sys.stderr.write(f"WARNING: {warning}\n")


def main() -> None:
    """Console entry point for analyze-pocket-charge."""
    parser = argparse.ArgumentParser(
        description="Measure net charge at the DnaK/Hsp70 SBD peptide-binding pocket.",
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        help="AlphaFold PDB file, or directory of PDB files with --batch",
    )
    parser.add_argument(
        "--reference-pocket",
        type=Path,
        default=_default_pocket_ref(),
        help="Pocket definition YAML (default: data/pocket_refs/dnak_sbd_pocket.yaml)",
    )
    parser.add_argument(
        "--reference-structure",
        type=Path,
        default=None,
        help="Reference AlphaFold PDB for alignment (default: AF-<ref accession>-... under data/dev_structures/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (single structure mode)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat structure argument as a directory; write one JSON per PDB",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pocket_charge_results"),
        help="Output directory for --batch and --merge-results modes",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default=None,
        help="Only include results at or above this combined confidence tier in CSV/summary output",
    )
    parser.add_argument(
        "--min-mapping-confidence",
        choices=["high", "medium", "low"],
        default=None,
        help="Only include results at or above this mapping confidence tier in CSV/summary output",
    )
    parser.add_argument(
        "--merge-results",
        type=Path,
        metavar="DIR",
        default=None,
        help="Merge existing pocket_charge_*.json files in DIR into summary CSV/JSON",
    )
    parser.add_argument(
        "--export-pocket-residues",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-residue pocket membership CSV for PyMOL validation (single structure)",
    )
    parser.add_argument(
        "--export-contact-attribution",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-contact residue charge attribution CSV (single structure)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Sweep shell radius, min_plddt, and SBD window; write stability report",
    )
    args = parser.parse_args()

    min_confidence: ConfidenceTier | None = args.min_confidence
    min_mapping_confidence: ConfidenceTier | None = args.min_mapping_confidence

    if args.merge_results is not None:
        _handle_merge_results(
            args,
            min_confidence=min_confidence,
            min_mapping_confidence=min_mapping_confidence,
        )
        return

    if args.structure is None:
        parser.error("structure path is required (unless using --merge-results)")

    ref_structure = args.reference_structure or _default_reference_structure(args.reference_pocket)

    if not args.reference_pocket.is_file():
        parser.error(f"Pocket reference not found: {args.reference_pocket}")
    if not ref_structure.is_file():
        parser.error(f"Reference structure not found: {ref_structure}")

    if args.sensitivity:
        _handle_sensitivity(args, ref_structure)
        return

    if args.batch:
        _handle_batch(
            args,
            ref_structure,
            min_confidence=min_confidence,
            min_mapping_confidence=min_mapping_confidence,
        )
        return

    _handle_single_structure(
        args,
        ref_structure,
        min_confidence=min_confidence,
        min_mapping_confidence=min_mapping_confidence,
    )


if __name__ == "__main__":
    main()
