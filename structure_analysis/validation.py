"""Experimental structure validation for pocket definitions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from structure_analysis.pdb_io import ResidueRecord, _open_pdb

if TYPE_CHECKING:
    from pathlib import Path


def load_pdb_chain(path: Path, chain_id: str) -> list[ResidueRecord]:
    """Load CA atoms for a single chain from a PDB file."""
    return _load_chain_residues(path, chain_id)


def _load_chain_residues(path: Path, chain_id: str) -> list[ResidueRecord]:
    residues: dict[tuple[int, str], ResidueRecord] = {}
    index = 0
    with _open_pdb(path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            resname = line[17:20].strip().upper()
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            try:
                plddt = float(line[60:66])
            except ValueError:
                plddt = None
            key = (resnum, resname)
            if key not in residues:
                index += 1
                residues[key] = ResidueRecord(
                    index=index,
                    resnum=resnum,
                    resname=resname,
                    ca_coord=(x, y, z),
                    plddt=plddt,
                )
    return list(residues.values())


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def derive_peptide_contact_resnums(
    protein_residues: list[ResidueRecord],
    peptide_residues: list[ResidueRecord],
    radius_angstrom: float = 8.0,
) -> list[int]:
    """Return sorted protein residue numbers with CA within radius of any peptide CA."""
    peptide_coords = [r.ca_coord for r in peptide_residues if r.ca_coord is not None]
    contacts: set[int] = set()
    for residue in protein_residues:
        if residue.ca_coord is None:
            continue
        for pep_coord in peptide_coords:
            if _distance(residue.ca_coord, pep_coord) <= radius_angstrom:
                contacts.add(residue.resnum)
                break
    return sorted(contacts)
