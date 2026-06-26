"""Load pocket reference definitions from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class PocketReference:
    """Literature-anchored pocket definition on the reference protein."""

    pocket_definition: str
    reference_accession: str
    reference_pdb: str
    reference_pdb_url: str
    shell_radius_angstrom: float
    sbd_residue_range: tuple[int, int]
    min_plddt: float
    max_contact_ca_distance: float
    contact_residues: tuple[int, ...]


def load_pocket_reference(path: Path) -> PocketReference:
    """Load a pocket reference YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sbd_range = data.get("sbd_residue_range", [386, 638])
    return PocketReference(
        pocket_definition=str(data["pocket_definition"]),
        reference_accession=str(data["reference_accession"]),
        reference_pdb=str(data["reference_pdb"]),
        reference_pdb_url=str(data.get("reference_pdb_url", "")),
        shell_radius_angstrom=float(data.get("shell_radius_angstrom", 8.0)),
        sbd_residue_range=(int(sbd_range[0]), int(sbd_range[1])),
        min_plddt=float(data.get("min_plddt", 70.0)),
        max_contact_ca_distance=float(data.get("max_contact_ca_distance", 5.0)),
        contact_residues=tuple(int(r) for r in data["contact_residues"]),
    )
