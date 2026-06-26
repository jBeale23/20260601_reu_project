"""Load AlphaFold PDB structures and extract per-residue metadata."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from structure_analysis.constants import AA3_TO_1

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    """One protein residue from a PDB file."""

    index: int  # 1-based position in extracted sequence (contiguous)
    resnum: int  # PDB residue number (may have gaps)
    resname: str  # three-letter code
    ca_coord: tuple[float, float, float] | None
    plddt: float | None  # AlphaFold confidence from B-factor column


def _open_pdb(path: Path):  # noqa: ANN202
    if path.suffix == ".gz" or path.name.endswith(".pdb.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def parse_accession_from_path(path: Path) -> str | None:
    """Extract UniProt accession from AlphaFold filename AF-<accession>-F1-..."""
    match = re.search(r"AF-([A-Z0-9]+)-F\d+", path.name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_structure_residues(path: Path) -> list[ResidueRecord]:
    """Parse ATOM records and return one record per residue (first altloc)."""
    residues: dict[tuple[str, int, str], ResidueRecord] = {}
    index = 0

    with _open_pdb(path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
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
            key = (chain, resnum, resname)
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


def residues_to_sequence(residues: list[ResidueRecord]) -> str:
    """Build a one-letter sequence from parsed residues."""
    chars: list[str] = []
    for residue in residues:
        one = AA3_TO_1.get(residue.resname)
        if one is None:
            one = "X"
        chars.append(one)
    return "".join(chars)


def residue_by_index(residues: list[ResidueRecord]) -> dict[int, ResidueRecord]:
    """Map 1-based sequence index to residue record."""
    return {r.index: r for r in residues}


def indices_in_sbd_window(
    residues: list[ResidueRecord],
    sbd_start_resnum: int,
    sbd_end_resnum: int,
) -> set[int]:
    """Return 1-based indices whose PDB residue number falls in the SBD window."""
    return {r.index for r in residues if sbd_start_resnum <= r.resnum <= sbd_end_resnum}
