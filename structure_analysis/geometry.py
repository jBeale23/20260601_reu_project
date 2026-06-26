"""Structure-guided geometry for pocket mapping validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from structure_analysis.pdb_io import residue_by_index

if TYPE_CHECKING:
    from structure_analysis.pdb_io import ResidueRecord

MIN_SBD_PAIRS_FOR_SUPERPOSITION = 30
RELIABLE_SUPERPOSITION_RMSD = 10.0
KABSCH_MIN_PAIRED_COORDS = 3
MIN_DISTANCE_MATRIX_COORDS = 2


@dataclass(frozen=True, slots=True)
class StructuralMetrics:
    """Structural validation metrics for mapped pocket contacts."""

    mean_contact_ca_distance: float | None
    max_contact_ca_distance: float | None
    n_contacts_within_threshold: int
    contact_ca_distances: dict[int, float]
    n_superposition_pairs: int
    sbd_superposition_rmsd: float | None
    contact_drmsd: float | None


def kabsch_superpose(
    ref_coords: np.ndarray,
    tgt_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation matrix R and translation t mapping target onto reference."""
    if ref_coords.shape != tgt_coords.shape or ref_coords.shape[0] < KABSCH_MIN_PAIRED_COORDS:
        msg = "Kabsch requires at least 3 paired 3D coordinates"
        raise ValueError(msg)

    ref_centroid = ref_coords.mean(axis=0)
    tgt_centroid = tgt_coords.mean(axis=0)
    ref_centered = ref_coords - ref_centroid
    tgt_centered = tgt_coords - tgt_centroid

    covariance = ref_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = ref_centroid - rotation @ tgt_centroid
    return rotation, translation


def apply_transform(
    coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply rotation and translation to Nx3 coordinates."""
    return (rotation @ coords.T).T + translation


def _collect_sbd_ca_pairs(
    reference_residues: list[ResidueRecord],
    target_residues: list[ResidueRecord],
    ref_index_to_target_index: dict[int, int],
    sbd_target_indices: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    ref_by_index = residue_by_index(reference_residues)
    tgt_by_index = residue_by_index(target_residues)
    ref_coords: list[tuple[float, float, float]] = []
    tgt_coords: list[tuple[float, float, float]] = []

    for ref_index, tgt_index in ref_index_to_target_index.items():
        if tgt_index not in sbd_target_indices:
            continue
        ref_residue = ref_by_index.get(ref_index)
        tgt_residue = tgt_by_index.get(tgt_index)
        if ref_residue is None or tgt_residue is None:
            continue
        if ref_residue.ca_coord is None or tgt_residue.ca_coord is None:
            continue
        ref_coords.append(ref_residue.ca_coord)
        tgt_coords.append(tgt_residue.ca_coord)

    if not ref_coords:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(ref_coords, dtype=float), np.array(tgt_coords, dtype=float)


def _contact_coords(
    residues: list[ResidueRecord],
    resnum_to_index: dict[int, int],
    contact_resnums: list[int],
) -> list[np.ndarray]:
    by_index = residue_by_index(residues)
    coords: list[np.ndarray] = []
    for resnum in contact_resnums:
        index = resnum_to_index.get(resnum)
        if index is None:
            continue
        residue = by_index.get(index)
        if residue is None or residue.ca_coord is None:
            continue
        coords.append(np.array(residue.ca_coord, dtype=float))
    return coords


def _distance_matrix(coords: list[np.ndarray]) -> np.ndarray | None:
    if len(coords) < MIN_DISTANCE_MATRIX_COORDS:
        return None
    n = len(coords)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = float(np.linalg.norm(coords[i] - coords[j]))
    return matrix


def _contact_drmsd(
    reference_residues: list[ResidueRecord],
    target_residues: list[ResidueRecord],
    resnum_to_target_index: dict[int, int],
    contact_resnums: list[int],
) -> float | None:
    """Rotation-invariant pocket geometry similarity via pairwise Ca distance matrices."""
    ref_resnum_to_index = {r.resnum: r.index for r in reference_residues}
    ref_coords = _contact_coords(reference_residues, ref_resnum_to_index, contact_resnums)
    tgt_coords = _contact_coords(target_residues, resnum_to_target_index, contact_resnums)
    if len(ref_coords) != len(tgt_coords) or len(ref_coords) < MIN_DISTANCE_MATRIX_COORDS:
        return None
    ref_matrix = _distance_matrix(ref_coords)
    tgt_matrix = _distance_matrix(tgt_coords)
    if ref_matrix is None or tgt_matrix is None:
        return None
    diff = ref_matrix - tgt_matrix
    return float(np.sqrt(np.mean(diff**2)))


@dataclass(frozen=True, slots=True)
class StructuralMetricsInputs:
    """Structure-guided inputs for pocket contact geometry validation."""

    reference_residues: list[ResidueRecord]
    target_residues: list[ResidueRecord]
    ref_index_to_target_index: dict[int, int]
    resnum_to_target_index: dict[int, int]
    sbd_target_indices: set[int]
    contact_resnums: list[int]
    max_contact_ca_distance: float = 5.0
    min_superposition_pairs: int = MIN_SBD_PAIRS_FOR_SUPERPOSITION


def compute_structural_metrics(inputs: StructuralMetricsInputs) -> StructuralMetrics:
    """Superpose target SBD onto reference; report Ca distances and DRMSD."""
    ref_pairs, tgt_pairs = _collect_sbd_ca_pairs(
        inputs.reference_residues,
        inputs.target_residues,
        inputs.ref_index_to_target_index,
        inputs.sbd_target_indices,
    )
    n_pairs = len(ref_pairs)
    contact_drmsd = _contact_drmsd(
        inputs.reference_residues,
        inputs.target_residues,
        inputs.resnum_to_target_index,
        inputs.contact_resnums,
    )
    empty = StructuralMetrics(None, None, 0, {}, n_pairs, None, contact_drmsd)

    if n_pairs < inputs.min_superposition_pairs:
        return empty

    rotation, translation = kabsch_superpose(ref_pairs, tgt_pairs)
    aligned_pairs = apply_transform(tgt_pairs, rotation, translation)
    sbd_rmsd = float(np.sqrt(np.mean(np.sum((ref_pairs - aligned_pairs) ** 2, axis=1))))

    ref_by_index = residue_by_index(inputs.reference_residues)
    tgt_by_index = residue_by_index(inputs.target_residues)
    ref_resnum_to_index = {r.resnum: r.index for r in inputs.reference_residues}

    contact_distances: dict[int, float] = {}
    for resnum, tgt_index in inputs.resnum_to_target_index.items():
        ref_index = ref_resnum_to_index.get(resnum)
        if ref_index is None:
            continue
        ref_residue = ref_by_index.get(ref_index)
        tgt_residue = tgt_by_index.get(tgt_index)
        if ref_residue is None or tgt_residue is None:
            continue
        if ref_residue.ca_coord is None or tgt_residue.ca_coord is None:
            continue
        tgt_transformed = apply_transform(
            np.array([tgt_residue.ca_coord], dtype=float),
            rotation,
            translation,
        )[0]
        ref_coord = np.array(ref_residue.ca_coord, dtype=float)
        distance = float(np.linalg.norm(ref_coord - tgt_transformed))
        contact_distances[resnum] = distance

    if not contact_distances:
        return StructuralMetrics(
            None,
            None,
            0,
            {},
            n_pairs,
            sbd_rmsd,
            contact_drmsd,
        )

    distances = list(contact_distances.values())
    n_within = sum(1 for d in distances if d <= inputs.max_contact_ca_distance)
    return StructuralMetrics(
        mean_contact_ca_distance=sum(distances) / len(distances),
        max_contact_ca_distance=max(distances),
        n_contacts_within_threshold=n_within,
        contact_ca_distances=contact_distances,
        n_superposition_pairs=n_pairs,
        sbd_superposition_rmsd=sbd_rmsd,
        contact_drmsd=contact_drmsd,
    )
