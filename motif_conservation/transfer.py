"""Held-out DnaK transfer / charge-inversion sanity checks."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def load_charge_inversion_accessions(pocket_csv: Path) -> set[str]:
    """Load accessions flagged charge_inversion_candidate from pocket summary CSV."""
    with pocket_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        flagged: set[str] = set()
        for row in reader:
            accession = (row.get("accession") or "").strip()
            raw = (row.get("charge_inversion_candidate") or "").strip().lower()
            if accession and raw in {"1", "true", "yes", "y"}:
                flagged.add(accession)
        return flagged


def load_motif_accessions(motif_csv: Path, *, domain_family: str | None = None) -> set[str]:
    """Load accessions from motif_accession_features.csv, optionally filtered by family."""
    with motif_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        accessions: set[str] = set()
        for row in reader:
            if domain_family and row.get("domain_family") != domain_family:
                continue
            accession = (row.get("accession") or "").strip()
            if accession:
                accessions.add(accession)
        return accessions


def held_out_overlap_report(
    motif_csv: Path,
    pocket_csv: Path,
    *,
    domain_family: str | None = None,
) -> dict[str, object]:
    """Compare motif-analyzed accessions to known charge-inversion candidates.

    This is a **held-out sanity check** only — window lengths must not be optimized
    against charge-inversion labels.
    """
    motif_ids = load_motif_accessions(motif_csv, domain_family=domain_family)
    inversion_ids = load_charge_inversion_accessions(pocket_csv)
    overlap = sorted(motif_ids & inversion_ids)
    return {
        "n_motif_accessions": len(motif_ids),
        "n_charge_inversion_candidates": len(inversion_ids),
        "n_overlap": len(overlap),
        "overlap_accessions": overlap,
        "domain_family_filter": domain_family,
        "note": ("Held-out sanity check only; do not optimize window length on charge-inversion labels."),
    }
