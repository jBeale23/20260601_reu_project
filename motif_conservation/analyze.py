"""Orchestrate DnaJ domain-family motif / charge-window analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from motif_conservation.alignment_frames import AlignedFamily, align_domain_family
from motif_conservation.charge_alphabet import sequence_net_charge
from motif_conservation.constants import (
    DEFAULT_WINDOW_LENGTHS,
    DOMAIN_FAMILY_LABELS,
    IDR_LIKE_DOMAIN_PFAMS,
    MIN_ALIGNED_COLUMNS,
    MIN_FAMILY_MEMBERS,
    MOTIF_ACCESSION_COLUMNS,
    MOTIF_SUMMARY_COLUMNS,
)
from motif_conservation.families import (
    DomainSlice,
    dedupe_slices_by_accession,
    group_domain_families,
    load_dnaj_proteins,
)
from motif_conservation.idr_blocks import block_grammar, grammar_recurrence, segment_composition_blocks
from motif_conservation.windows import WindowSweepResult, per_sequence_window_charge, sweep_window_lengths

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class FamilyMotifResult:
    """Per-family conservation summary."""

    domain_family: str
    n_members: int
    n_aligned_columns: int
    best_window_length: int
    best_conservation_score: float
    mean_net_charge_at_best: float
    preferred_block_grammar: str
    scores_by_length: dict[int, float]


def analyze_family(
    domain_family: str,
    slices: list[DomainSlice],
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    min_members: int = MIN_FAMILY_MEMBERS,
) -> tuple[FamilyMotifResult | None, list[dict[str, Any]], AlignedFamily | None]:
    """Analyze one domain family; return summary, per-accession rows, and alignment."""
    unique = dedupe_slices_by_accession(slices)
    if len(unique) < min_members:
        return None, [], None

    is_idr = any(item.pfam in IDR_LIKE_DOMAIN_PFAMS for item in unique)
    grammars = [block_grammar(segment_composition_blocks(item.sequence)) for item in unique]
    preferred_grammar = grammar_recurrence(grammars)

    aligned: AlignedFamily | None = None
    sweep: WindowSweepResult | None = None
    if not is_idr:
        aligned = align_domain_family(domain_family, unique)
        if aligned.aligned_sequences and len(aligned.aligned_sequences[0]) >= MIN_ALIGNED_COLUMNS:
            sweep = sweep_window_lengths(aligned, window_lengths=window_lengths)

    best_window = sweep.best_window_length if sweep else 0
    best_score = sweep.best_conservation_score if sweep else 0.0
    mean_charge = sweep.mean_charge_by_length.get(best_window, 0.0) if sweep else 0.0
    n_cols = sweep.n_aligned_columns if sweep else 0
    scores = sweep.scores_by_length if sweep else {}

    summary = FamilyMotifResult(
        domain_family=domain_family,
        n_members=len(unique),
        n_aligned_columns=n_cols,
        best_window_length=best_window,
        best_conservation_score=best_score,
        mean_net_charge_at_best=mean_charge,
        preferred_block_grammar=preferred_grammar,
        scores_by_length=scores,
    )

    accession_rows: list[dict[str, Any]] = []
    for item, grammar in zip(unique, grammars, strict=True):
        window_charge = per_sequence_window_charge(item.sequence, best_window) if best_window else None
        accession_rows.append(
            {
                "accession": item.accession,
                "domain_family": domain_family,
                "start": item.start,
                "end": item.end,
                "sequence_length": len(item.sequence),
                "net_charge": sequence_net_charge(item.sequence),
                "block_grammar": grammar,
                "best_window_length": best_window,
                "window_net_charge": "" if window_charge is None else f"{window_charge:.4f}",
            },
        )
    return summary, accession_rows, aligned


def analyze_dnaj_fetch(
    fetch_json: Path,
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    min_members: int = MIN_FAMILY_MEMBERS,
    max_per_family: int | None = None,
) -> tuple[list[FamilyMotifResult], list[dict[str, Any]], dict[str, Any]]:
    """Run motif conservation across all DnaJ domain families in a fetch JSON."""
    proteins = load_dnaj_proteins(fetch_json)
    grouped = group_domain_families(proteins)

    summaries: list[FamilyMotifResult] = []
    accession_rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}

    for domain_family, slices in sorted(grouped.items()):
        limited = slices if max_per_family is None else slices[:max_per_family]
        summary, rows, _aligned = analyze_family(
            domain_family,
            limited,
            window_lengths=window_lengths,
            min_members=min_members,
        )
        if summary is None:
            continue
        summaries.append(summary)
        accession_rows.extend(rows)
        curves[domain_family] = {
            "scores_by_length": {str(k): v for k, v in summary.scores_by_length.items()},
            "best_window_length": summary.best_window_length,
            "best_conservation_score": summary.best_conservation_score,
            "preferred_block_grammar": summary.preferred_block_grammar,
            "n_members": summary.n_members,
        }

    return summaries, accession_rows, curves


def write_outputs(
    output_dir: Path,
    summaries: list[FamilyMotifResult],
    accession_rows: list[dict[str, Any]],
    curves: dict[str, Any],
) -> None:
    """Write summary CSV, per-accession CSV, and conservation curves JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "motif_family_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOTIF_SUMMARY_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "domain_family": summary.domain_family,
                    "n_members": summary.n_members,
                    "n_aligned_columns": summary.n_aligned_columns,
                    "best_window_length": summary.best_window_length,
                    "best_conservation_score": f"{summary.best_conservation_score:.6f}",
                    "mean_net_charge_at_best": f"{summary.mean_net_charge_at_best:.6f}",
                    "preferred_block_grammar": summary.preferred_block_grammar,
                },
            )

    accession_path = output_dir / "motif_accession_features.csv"
    with accession_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOTIF_ACCESSION_COLUMNS)
        writer.writeheader()
        for row in accession_rows:
            writer.writerow({column: row.get(column, "") for column in MOTIF_ACCESSION_COLUMNS})

    curves_path = output_dir / "motif_conservation_curves.json"
    curves_path.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")

    meta = {
        "n_families": len(summaries),
        "n_accession_rows": len(accession_rows),
        "domain_family_labels": DOMAIN_FAMILY_LABELS,
    }
    (output_dir / "motif_run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def family_result_to_dict(result: FamilyMotifResult) -> dict[str, Any]:
    """Serialize a family result."""
    return asdict(result)
