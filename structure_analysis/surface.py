"""Solvent accessibility and pocket geometry from an all-atom structure.

The features in :mod:`structure_analysis.features` work from alpha carbons alone, which is
enough for compactness and topology but not for anything about a *surface*. Whether a
charged residue is exposed, and whether a concavity is deep enough to bind something, are
all-atom questions.

Solvent accessible surface area
-------------------------------
Computed by the Shrake-Rupley algorithm: roll a probe of water radius over the van der
Waals surface by placing test points on a sphere around each atom and counting how many
escape every other atom. The fraction that escape, times the sphere area, is that atom's
accessible area.

Accessibility is reported *relative* to the residue's maximum - its area as a free
tripeptide - because absolute area scales with residue size, so a fully exposed glycine
and a buried tryptophan can report the same square angstroms. The relative value is the
one that means "exposed".

Pocket detection
----------------
A LIGSITE-style grid scan. The structure is embedded in a lattice; each empty grid point is
probed along the three axes and the four cubic diagonals, and a scan line that leaves the
point, meets protein, and returns is a *protein-solvent-protein event*. A point enclosed on
many axes sits in a concavity rather than on an open face. Clusters of such points are
pockets.

This is chosen over an alpha-sphere method (fpocket and relatives) because it needs no
Delaunay triangulation and its one parameter - how many enclosing directions are required -
maps directly onto a geometric statement about the site. It is less sensitive to shallow
grooves than alpha-sphere methods, so the pockets it reports are the deep ones; the count
should be read as a floor.

Each pocket carries the charge and the accessibility of its lining residues, which is the
combination that matters: a deep cavity lined with exposed charged residues is a binding
site, while the same cavity lined with buried hydrophobics is a folding core.

Two filters keep the output honest, both added after the first validation run produced a
1,675-cubic-angstrom "pocket" lined by eleven residues at pLDDT 51 - the gap between two
domains of a multi-domain protein, which passes every enclosure test because the domains
surround it. A cavity must have walls in proportion to its size, and its surroundings must
be placed confidently enough for the geometry to mean anything.
"""

from __future__ import annotations

import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

# Van der Waals radii in angstroms, by element. Bondi values, which are the ones the
# Shrake-Rupley literature uses.
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
_DEFAULT_VDW = 1.70

# Radius of a water molecule. The surface traced is where a water centre can reach.
PROBE_RADIUS = 1.4

# Test points per atom. Shrake-Rupley's accuracy improves as the square root of this;
# 92 is the classic value and holds the error near 1-2%, which is far below the
# residue-to-residue variation being measured.
SPHERE_POINTS = 92

# Maximum accessible area per residue as a free Gly-X-Gly tripeptide, in square angstroms
# (Tien et al. 2013, theoretical values). Dividing by these converts absolute area into
# the relative accessibility that actually means "exposed".
_MAX_ASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}
_DEFAULT_MAX_ASA = 200.0

# Relative accessibility above which a residue counts as exposed. The conventional cut in
# the accessibility literature.
EXPOSED_RSA = 0.25

# Grid spacing for pocket detection, in angstroms. One angstrom resolves a cavity able to
# hold a small molecule without making the lattice unaffordable for a large protein.
GRID_SPACING = 1.0

# Enclosing directions, out of seven scanned, required for a grid point to count as buried
# in a concavity. LIGSITE's own threshold.
MIN_ENCLOSURE = 5

# Grid points required before a cluster is called a pocket. At 1 A spacing this is a
# volume floor of 30 cubic angstroms, below which a "pocket" is a surface dimple.
MIN_POCKET_POINTS = 30

# Lattice cells above which a structure is skipped rather than risk exhausting a node's
# memory on one unusually large model.
MAX_GRID_CELLS = 40_000_000

# Pocket points handled per pass when measuring which residues line a cavity.
#
# The natural expression - one array of every point against every atom - allocates
# n_points x n_atoms x 3 doubles. For a large protein that is 38 GB, and with one worker
# per core it exhausted a 180 GB node in five minutes. Chunking bounds the peak to roughly
# CHUNK x n_atoms x 8 bytes regardless of cavity size.
LINING_CHUNK = 2048

# Minimum lining residues per 100 cubic angstroms. A real binding site has walls: a
# 400-cubic-angstrom pocket is typically lined by 15-25 residues, giving 4-6 per 100. Space
# between two domains passes every enclosure test and has almost none - the first
# validation run reported a 1,675-cubic-angstrom "pocket" lined by 11 residues, which is
# the gap between the J-domain and the C-terminal domain, not a cavity.
MIN_LINING_DENSITY = 1.5

# Mean pLDDT below which a detected cavity is discarded. AlphaFold placing the surrounding
# atoms with very low confidence means the cavity is an artefact of where an unplaceable
# linker happened to land. This is the same threshold used for disorder elsewhere.
MIN_POCKET_PLDDT = 50.0

_POSITIVE = {"LYS", "ARG"}
_NEGATIVE = {"ASP", "GLU"}


@dataclass(frozen=True, slots=True)
class Atom:
    """One atom, with the residue it belongs to."""

    resnum: int
    resname: str
    name: str
    element: str
    coord: tuple[float, float, float]
    plddt: float | None

    @property
    def radius(self) -> float:
        """Van der Waals radius for this atom's element."""
        return _VDW_RADII.get(self.element, _DEFAULT_VDW)


def load_atoms(path: Path) -> list[Atom]:
    """Read every protein atom from a PDB file, gzipped or plain."""
    opener = gzip.open if path.name.endswith(".gz") else open
    atoms: list[Atom] = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            element = line[76:78].strip() or line[12:16].strip()[:1]
            if element == "H":
                # AlphaFold models carry no hydrogens, but a PDB entry may; they are
                # excluded because the max-ASA reference values are heavy-atom surfaces.
                continue
            try:
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                resnum = int(line[22:26])
            except ValueError:
                continue
            plddt: float | None
            try:
                plddt = float(line[60:66])
            except ValueError:
                plddt = None
            atoms.append(
                Atom(
                    resnum=resnum,
                    resname=line[17:20].strip(),
                    name=line[12:16].strip(),
                    element=element,
                    coord=coord,
                    plddt=plddt,
                ),
            )
    return atoms


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` roughly equidistant points on the unit sphere.

    A golden-angle spiral rather than a random draw, so the estimate is deterministic and
    its error does not vary between runs of the same structure.
    """
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
    )


def atom_sasa(atoms: Sequence[Atom], *, n_points: int = SPHERE_POINTS) -> np.ndarray:
    """Solvent accessible surface area per atom, in square angstroms."""
    if not atoms:
        return np.zeros(0)

    coords = np.array([atom.coord for atom in atoms], dtype=np.float64)
    radii = np.array([atom.radius for atom in atoms], dtype=np.float64) + PROBE_RADIUS
    sphere = _fibonacci_sphere(n_points)

    # Only atoms whose expanded spheres could overlap can occlude one another, so the
    # neighbour list keeps this near-linear instead of N^2 in the inner loop.
    max_reach = float(radii.max()) * 2.0
    areas = np.zeros(len(atoms), dtype=np.float64)
    for index in range(len(atoms)):
        offsets = coords - coords[index]
        distances = np.sqrt((offsets**2).sum(axis=1))
        near = np.flatnonzero((distances < max_reach) & (distances > 0.0))
        if near.size == 0:
            areas[index] = 4.0 * math.pi * radii[index] ** 2
            continue

        test = coords[index] + sphere * radii[index]
        # A test point is occluded if it falls inside any neighbour's expanded sphere.
        deltas = test[:, None, :] - coords[near][None, :, :]
        squared = (deltas**2).sum(axis=-1)
        occluded = (squared < (radii[near] ** 2)[None, :]).any(axis=1)
        accessible = float((~occluded).sum()) / n_points
        areas[index] = accessible * 4.0 * math.pi * radii[index] ** 2
    return areas


@dataclass(frozen=True, slots=True)
class ResidueSurface:
    """Accessibility and charge for one residue."""

    resnum: int
    resname: str
    absolute_sasa: float
    relative_sasa: float
    charge: float
    mean_plddt: float | None

    @property
    def is_exposed(self) -> bool:
        """Whether the residue presents a meaningful share of its surface to solvent."""
        return self.relative_sasa >= EXPOSED_RSA


def residue_surfaces(atoms: Sequence[Atom], areas: np.ndarray) -> list[ResidueSurface]:
    """Aggregate per-atom areas into per-residue accessibility."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        grouped[atom.resnum].append(index)

    surfaces: list[ResidueSurface] = []
    for resnum in sorted(grouped):
        indices = grouped[resnum]
        resname = atoms[indices[0]].resname
        absolute = float(areas[indices].sum())
        plddts = [atoms[i].plddt for i in indices if atoms[i].plddt is not None]
        surfaces.append(
            ResidueSurface(
                resnum=resnum,
                resname=resname,
                absolute_sasa=absolute,
                relative_sasa=absolute / _MAX_ASA.get(resname, _DEFAULT_MAX_ASA),
                charge=1.0 if resname in _POSITIVE else -1.0 if resname in _NEGATIVE else 0.0,
                mean_plddt=(sum(plddts) / len(plddts)) if plddts else None,
            ),
        )
    return surfaces


@dataclass(frozen=True, slots=True)
class Pocket:
    """One detected cavity, with the chemistry of its lining."""

    rank: int
    volume: float
    depth: float
    n_lining_residues: int
    net_charge: float
    mean_relative_sasa: float
    fraction_exposed_lining: float
    mean_plddt: float | None

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "rank": self.rank,
            "volume": round(self.volume, 1),
            "depth": round(self.depth, 2),
            "n_lining_residues": self.n_lining_residues,
            "net_charge": round(self.net_charge, 2),
            "mean_relative_sasa": round(self.mean_relative_sasa, 4),
            "fraction_exposed_lining": round(self.fraction_exposed_lining, 4),
            "mean_plddt": round(self.mean_plddt, 2) if self.mean_plddt is not None else None,
        }


_SCAN_DIRECTIONS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
)


def _enclosure_counts(occupied: np.ndarray) -> np.ndarray:
    """For each empty lattice point, how many of seven scan axes enclose it.

    A direction encloses a point when the scan line meets protein on both sides - the
    protein-solvent-protein event LIGSITE is built on.
    """
    counts = np.zeros(occupied.shape, dtype=np.int8)
    for direction in _SCAN_DIRECTIONS:
        before = np.zeros(occupied.shape, dtype=bool)
        after = np.zeros(occupied.shape, dtype=bool)
        # Cumulative "has met protein" in each sense along the axis.
        shifted = occupied
        for _ in range(max(occupied.shape)):
            shifted = _shift(shifted, direction)
            before |= shifted
            if not shifted.any():
                break
        shifted = occupied
        opposite = tuple(-value for value in direction)
        for _ in range(max(occupied.shape)):
            shifted = _shift(shifted, opposite)
            after |= shifted
            if not shifted.any():
                break
        counts += (before & after).astype(np.int8)
    return counts


def _shift(grid: np.ndarray, direction: tuple[int, int, int]) -> np.ndarray:
    """Translate a boolean lattice by one step, filling vacated cells with False."""
    result = grid
    for axis, step in enumerate(direction):
        if step == 0:
            continue
        result = np.roll(result, step, axis=axis)
        index: list[slice | int] = [slice(None)] * 3
        index[axis] = 0 if step > 0 else -1
        result[tuple(index)] = False
    return result


def _cluster(points: np.ndarray) -> Iterator[np.ndarray]:
    """Group lattice points into connected components by flood fill."""
    remaining = {tuple(point) for point in points}
    neighbours = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)
    ]
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for offset in neighbours:
                candidate = (current[0] + offset[0], current[1] + offset[1], current[2] + offset[2])
                if candidate in remaining:
                    remaining.discard(candidate)
                    stack.append(candidate)
                    component.append(candidate)
        yield np.array(component)


def _lining_and_depth(
    centres: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Which atoms line a cavity, and how far it reaches from the nearest atom.

    Chunked over pocket points, since the whole-cavity array would be tens of gigabytes
    for a large protein; see :data:`LINING_CHUNK`.
    """
    cutoff_squared = (radii + 2 * PROBE_RADIUS) ** 2
    touches = np.zeros(coords.shape[0], dtype=bool)
    deepest = 0.0
    for start in range(0, centres.shape[0], LINING_CHUNK):
        block = centres[start : start + LINING_CHUNK]
        squared = ((block[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
        touches |= (squared <= cutoff_squared[None, :]).any(axis=0)
        deepest = max(deepest, float(np.sqrt(squared.min(axis=1)).max()))
    return touches, deepest


def find_pockets(
    atoms: Sequence[Atom],
    surfaces: Sequence[ResidueSurface],
    *,
    spacing: float = GRID_SPACING,
    min_enclosure: int = MIN_ENCLOSURE,
    max_pockets: int = 5,
) -> list[Pocket]:
    """Detect cavities and describe the chemistry lining each one."""
    if not atoms:
        return []

    coords = np.array([atom.coord for atom in atoms], dtype=np.float64)
    radii = np.array([atom.radius for atom in atoms], dtype=np.float64)

    margin = 3.0
    low = coords.min(axis=0) - margin
    high = coords.max(axis=0) + margin
    dims = np.maximum(np.ceil((high - low) / spacing).astype(int) + 1, 1)
    if int(np.prod(dims)) > MAX_GRID_CELLS:
        # A lattice this large means an unusually big model; skip rather than exhaust the
        # node's memory on one protein.
        return []

    occupied = np.zeros(tuple(dims), dtype=bool)
    indices = np.floor((coords - low) / spacing).astype(int)
    reach = np.ceil((radii + PROBE_RADIUS) / spacing).astype(int)
    for centre, span in zip(indices, reach, strict=True):
        slices = tuple(
            slice(max(0, centre[axis] - span), min(dims[axis], centre[axis] + span + 1)) for axis in range(3)
        )
        occupied[slices] = True

    counts = _enclosure_counts(occupied)
    buried = np.argwhere((~occupied) & (counts >= min_enclosure))
    if buried.size == 0:
        return []

    by_resnum = {surface.resnum: surface for surface in surfaces}
    cell_volume = spacing**3
    pockets: list[Pocket] = []
    for component in _cluster(buried):
        if len(component) < MIN_POCKET_POINTS:
            continue
        centres = low + component * spacing
        # Lining residues: any atom within a probe diameter of a pocket point. Computed in
        # chunks, since the whole-cavity array would be tens of gigabytes (see LINING_CHUNK).
        touches, deepest = _lining_and_depth(centres, coords, radii)
        lining = {atoms[i].resnum for i in np.flatnonzero(touches)}
        residues = [by_resnum[num] for num in lining if num in by_resnum]
        if not residues:
            continue

        plddts = [item.mean_plddt for item in residues if item.mean_plddt is not None]
        volume = len(component) * cell_volume
        mean_plddt = (sum(plddts) / len(plddts)) if plddts else None

        # A cavity with too few walls for its size is the space between domains, and one
        # whose surroundings AlphaFold could not place is an artefact of that failure.
        if len(residues) / (volume / 100.0) < MIN_LINING_DENSITY:
            continue
        if mean_plddt is not None and mean_plddt < MIN_POCKET_PLDDT:
            continue

        pockets.append(
            Pocket(
                rank=0,
                volume=volume,
                # Depth: the pocket point furthest from any protein atom, i.e. how far
                # into open space the cavity reaches.
                depth=deepest,
                n_lining_residues=len(residues),
                net_charge=sum(item.charge for item in residues),
                mean_relative_sasa=sum(item.relative_sasa for item in residues) / len(residues),
                fraction_exposed_lining=sum(1 for item in residues if item.is_exposed) / len(residues),
                mean_plddt=mean_plddt,
            ),
        )

    pockets.sort(key=lambda item: -item.volume)
    return [
        Pocket(
            rank=rank,
            volume=item.volume,
            depth=item.depth,
            n_lining_residues=item.n_lining_residues,
            net_charge=item.net_charge,
            mean_relative_sasa=item.mean_relative_sasa,
            fraction_exposed_lining=item.fraction_exposed_lining,
            mean_plddt=item.mean_plddt,
        )
        for rank, item in enumerate(pockets[:max_pockets], start=1)
    ]
