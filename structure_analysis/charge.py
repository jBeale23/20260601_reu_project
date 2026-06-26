"""Compute charge and hydrophobicity metrics for a pocket residue set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from structure_analysis.constants import (
    HISTIDINE_RESIDUE,
    HYDROPHOBIC_RESIDUES,
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
)

if TYPE_CHECKING:
    from structure_analysis.pdb_io import ResidueRecord

BURIAL_NEIGHBOR_RADIUS_ANGSTROM = 6.0
BURIAL_NEIGHBOR_THRESHOLD = 8


@dataclass(frozen=True, slots=True)
class PocketChargeMetrics:
    """Charge summary for a mapped pocket on one structure."""

    n_positive: int
    n_negative: int
    net_charge: int
    n_histidine: int
    net_charge_excluding_his: int
    n_hydrophobic: int
    mapped_pocket_residues: int
    mean_plddt_pocket: float | None
    n_low_plddt_excluded: int
    exposed_net_charge: int
    n_buried_charged: int


def _is_buried(
    residue: ResidueRecord,
    all_residues: list[ResidueRecord],
    radius: float = BURIAL_NEIGHBOR_RADIUS_ANGSTROM,
    threshold: int = BURIAL_NEIGHBOR_THRESHOLD,
) -> bool:
    if residue.ca_coord is None:
        return False
    coord = np.array(residue.ca_coord, dtype=float)
    neighbors = 0
    for other in all_residues:
        if other.index == residue.index or other.ca_coord is None:
            continue
        dist = np.linalg.norm(coord - np.array(other.ca_coord, dtype=float))
        if dist <= radius:
            neighbors += 1
    return neighbors >= threshold


def _charge_contribution(resname: str, *, include_his: bool = True) -> int:
    if resname in POSITIVE_RESIDUES:
        if resname == HISTIDINE_RESIDUE and not include_his:
            return 0
        return 1
    if resname in NEGATIVE_RESIDUES:
        return -1
    return 0


def _accumulate_pocket_residue(
    residue: ResidueRecord,
    all_residues: list[ResidueRecord],
    *,
    counts: dict[str, int],
    plddt_values: list[float],
) -> None:
    counts["mapped"] += 1
    if residue.plddt is not None:
        plddt_values.append(residue.plddt)
    name = residue.resname
    if name in POSITIVE_RESIDUES:
        counts["positive"] += 1
    if name == HISTIDINE_RESIDUE:
        counts["histidine"] += 1
    if name in NEGATIVE_RESIDUES:
        counts["negative"] += 1
    if name in HYDROPHOBIC_RESIDUES:
        counts["hydrophobic"] += 1

    charge = _charge_contribution(name, include_his=True)
    if charge != 0:
        if _is_buried(residue, all_residues):
            counts["buried_charged"] += 1
        else:
            counts["exposed_charge"] += charge


def compute_pocket_metrics(
    residues: list[ResidueRecord],
    pocket_indices: set[int],
    min_plddt: float = 0.0,
) -> PocketChargeMetrics:
    """Count charged and hydrophobic residues at the given 1-based indices."""
    by_index = {r.index: r for r in residues}
    counts = {
        "positive": 0,
        "negative": 0,
        "histidine": 0,
        "hydrophobic": 0,
        "mapped": 0,
        "low_plddt_excluded": 0,
        "exposed_charge": 0,
        "buried_charged": 0,
    }
    plddt_values: list[float] = []

    for index in sorted(pocket_indices):
        residue = by_index.get(index)
        if residue is None:
            continue
        if residue.plddt is not None and residue.plddt < min_plddt:
            counts["low_plddt_excluded"] += 1
            continue
        _accumulate_pocket_residue(residue, residues, counts=counts, plddt_values=plddt_values)

    net_charge = counts["positive"] - counts["negative"]
    net_excluding_his = net_charge - counts["histidine"]
    mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else None

    return PocketChargeMetrics(
        n_positive=counts["positive"],
        n_negative=counts["negative"],
        net_charge=net_charge,
        n_histidine=counts["histidine"],
        net_charge_excluding_his=net_excluding_his,
        n_hydrophobic=counts["hydrophobic"],
        mapped_pocket_residues=counts["mapped"],
        mean_plddt_pocket=mean_plddt,
        n_low_plddt_excluded=counts["low_plddt_excluded"],
        exposed_net_charge=counts["exposed_charge"],
        n_buried_charged=counts["buried_charged"],
    )


def residue_charge_contribution(resname: str) -> int:
    """Return +1, -1, or 0 for a residue three-letter code."""
    return _charge_contribution(resname, include_his=True)


def charge_density(net_charge: int, mapped_residues: int) -> float | None:
    """Net charge per counted pocket residue."""
    if mapped_residues == 0:
        return None
    return net_charge / mapped_residues
