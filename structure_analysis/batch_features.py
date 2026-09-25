"""Compute structural features for every protein in a domain store.

Pairs an accession list with a directory of AlphaFold models and produces one row per
protein. Proteins with no model are reported as missing rather than skipped silently: the
coverage figure is itself a result, since a systematic gap - say, models available for
model organisms and absent elsewhere - would bias every structural conclusion drawn from
the set.
"""

from __future__ import annotations

import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any

from domain_layout.progress import progress
from structure_analysis.features import compute_structural_features
from structure_analysis.pdb_io import load_structure_residues
from structure_analysis.surface import atom_sasa, find_pockets, load_atoms, residue_surfaces

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

STRUCTURE_FEATURE_COLUMNS = [
    "accession",
    "n_residues",
    "mean_plddt",
    "fraction_disordered_plddt",
    "fraction_confident_plddt",
    "radius_of_gyration",
    "compactness",
    "relative_contact_order",
    "fraction_buried",
    "surface_net_charge",
    "buried_net_charge",
    "surface_charge_density",
    "is_largely_disordered",
    # All-atom surface, which alpha carbons alone cannot give.
    "total_sasa",
    "mean_relative_sasa",
    "fraction_exposed",
    "exposed_net_charge",
    "buried_net_charge_sasa",
    "n_pockets",
    "largest_pocket_volume",
    "largest_pocket_charge",
    "largest_pocket_mean_rsa",
]

# Model file naming used by the AlphaFold bulk download, newest version first. Several
# versions coexist in a directory that has been topped up over time.
_MODEL_VERSIONS = (6, 5, 4, 3, 2)

PROGRESS_INTERVAL = 20000

# Same reasoning as the layout pipeline: worker processes each spawning a BLAS thread per
# core oversubscribes the allocation and degrades every other job on a shared node.
_THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def find_model(structures_dir: Path, accession: str) -> Path | None:
    """Locate the best available AlphaFold model for one accession."""
    for version in _MODEL_VERSIONS:
        for suffix in (".pdb.gz", ".pdb"):
            candidate = structures_dir / f"AF-{accession}-F1-model_v{version}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def features_for_accession(accession: str, structures_dir: Path) -> dict[str, Any] | None:
    """Structural features for one accession, or None when no usable model exists."""
    model = find_model(structures_dir, accession)
    if model is None:
        return None
    try:
        residues = load_structure_residues(model)
    except (OSError, ValueError):
        logger.warning("could not read %s", model, exc_info=True)
        return None

    features = compute_structural_features(accession, residues)
    if features is None:
        return None
    row = features.to_json_dict()

    # All-atom surface and cavity geometry. Failures here degrade the row rather than
    # discarding it: the alpha-carbon features are still valid without a surface.
    try:
        atoms = load_atoms(model)
        areas = atom_sasa(atoms)
        surfaces = residue_surfaces(atoms, areas)
        pockets = find_pockets(atoms, surfaces)
    except (OSError, ValueError, MemoryError):
        logger.warning("surface analysis failed for %s", accession, exc_info=True)
        return row

    if surfaces:
        exposed = [item for item in surfaces if item.is_exposed]
        row["total_sasa"] = round(float(areas.sum()), 1)
        row["mean_relative_sasa"] = round(sum(i.relative_sasa for i in surfaces) / len(surfaces), 4)
        row["fraction_exposed"] = round(len(exposed) / len(surfaces), 4)
        row["exposed_net_charge"] = round(sum(i.charge for i in exposed), 1)
        row["buried_net_charge_sasa"] = round(sum(i.charge for i in surfaces if not i.is_exposed), 1)

    row["n_pockets"] = len(pockets)
    if pockets:
        best = pockets[0]
        row["largest_pocket_volume"] = round(best.volume, 1)
        row["largest_pocket_charge"] = round(best.net_charge, 1)
        row["largest_pocket_mean_rsa"] = round(best.mean_relative_sasa, 4)
    return row


def _worker(chunk: Sequence[str], structures_dir: Path) -> list[dict[str, Any]]:
    return [row for accession in chunk if (row := features_for_accession(accession, structures_dir)) is not None]


@dataclass(frozen=True, slots=True)
class StructureRunSummary:
    """Coverage and aggregate statistics for one structural run."""

    n_requested: int
    n_with_model: int
    n_missing: int

    @property
    def coverage(self) -> float:
        """Fraction of requested proteins that had a usable model."""
        return self.n_with_model / self.n_requested if self.n_requested else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "n_requested": self.n_requested,
            "n_with_model": self.n_with_model,
            "n_missing_model": self.n_missing,
            "structure_coverage": round(self.coverage, 4),
        }


def compute_all(
    accessions: Sequence[str],
    structures_dir: Path,
    *,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], StructureRunSummary]:
    """Compute structural features for every accession that has a model."""
    if workers > 1:
        for name in _THREAD_LIMITS:
            os.environ.setdefault(name, "1")

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for index, accession in enumerate(progress(accessions, description="Structural features"), start=1):
            row = features_for_accession(accession, structures_dir)
            if row is not None:
                rows.append(row)
            if index % PROGRESS_INTERVAL == 0:
                logger.info("processed %s of %s", index, len(accessions))
    else:
        # More chunks than workers, so a slow chunk cannot strand a worker at the end and
        # progress is visible before the run finishes.
        size = max(1, len(accessions) // (workers * 8) or 1)
        chunks = [list(accessions[start : start + size]) for start in range(0, len(accessions), size)]
        context = get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_worker, chunk, structures_dir) for chunk in chunks]
            for future in as_completed(futures):
                rows.extend(future.result())
                done += 1
                logger.info("chunk %s of %s (%s protein(s) with models)", done, len(chunks), len(rows))

    rows.sort(key=lambda row: str(row["accession"]))
    summary = StructureRunSummary(
        n_requested=len(accessions),
        n_with_model=len(rows),
        n_missing=len(accessions) - len(rows),
    )
    return rows, summary


def write_structure_features(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write one row per modelled protein."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in STRUCTURE_FEATURE_COLUMNS})
