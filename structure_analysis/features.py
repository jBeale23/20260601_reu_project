"""Per-protein structural features from an AlphaFold model.

Everything else in this project is derived from sequence. That is a real limitation: the
routing decides which residues are disordered from a sequence predictor, the class rules
decide which domains are present from sequence-signature matches, and when a signature is
absent there is no way to tell "this protein lacks the domain" from "nothing annotated it".
A structure answers exactly that question, which is why it is the evidence the class
scheme needs and the one it has been missing.

What is computed, and why each one
----------------------------------
**pLDDT.** AlphaFold's per-residue confidence is not merely a quality flag - below about
50 it is a well-established predictor of intrinsic disorder, because the model cannot
place a residue that has no single place to be. That makes it an *independent* disorder
signal, derived from a different model on different training data than metapredict, so
disagreement between them localises exactly where the routing is least trustworthy.

**Burial.** A residue's neighbour count is a standard proxy for solvent accessibility.
Charge on the surface drives partner binding; the same charge buried is structural. Net
charge over a whole protein averages the two together and answers neither question.

**Compactness.** Radius of gyration against the value expected for a folded protein of
that length separates a compact domain from an extended or disordered chain, without
relying on a disorder predictor at all.

**Contact order.** The mean sequence separation of residues in spatial contact, normalised
by length. Low values mean local structure (helices); high values mean the fold brings
distant parts of the chain together. It distinguishes topology, which is invisible to
composition and to alignment alike.

Deliberate omissions
--------------------
Solvent accessibility is a neighbour count rather than a Shrake-Rupley or Lee-Richards
surface. The exact calculation needs all atoms and is far slower; the CA neighbour count
correlates strongly with relative accessibility and is the standard cheap substitute. It
is reported under a name that says what it is, and :func:`burial_from_neighbours` is
isolated so a true surface calculation can replace it without touching anything else.

Pocket geometry - depth, volume, allosteric site detection - is not here. Doing it
honestly needs a cavity-detection algorithm rather than a heuristic, and a wrong pocket
volume is worse than no pocket volume. :mod:`structure_analysis.charge` already computes
charge at a *reference* pocket defined from a solved structure, which is the defensible
version of the same idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from structure_analysis.pdb_io import ResidueRecord

# pLDDT below this is AlphaFold reporting that it cannot place the residue, which for a
# well-modelled proteome is the signature of intrinsic disorder rather than of a bad model.
# The threshold is AlphaFold's own published "very low" band.
DISORDER_PLDDT = 50.0

# The "confident" band. Between the two lies genuinely ambiguous territory, so both are
# reported rather than one cut being presented as the boundary.
CONFIDENT_PLDDT = 70.0

# Neighbour sphere for the burial proxy. Ten angstroms between alpha carbons is the
# conventional radius for contact-density measures; it spans roughly the first two
# coordination shells, which is what makes the count track accessibility.
NEIGHBOUR_RADIUS = 10.0

# CA-CA distance defining a contact for contact order. Eight angstroms is the standard
# choice and corresponds to side chains being able to touch.
CONTACT_DISTANCE = 8.0

# Contacts closer than this in sequence are excluded from contact order: neighbours along
# the backbone are in contact trivially and say nothing about the fold.
MIN_CONTACT_SEPARATION = 3

# Neighbour count at or above which a residue counts as buried, calibrated so that a
# typical globular domain reports roughly a third of its residues buried.
BURIED_NEIGHBOURS = 18

# Fraction of a chain below the very-low-confidence band before it reads as disordered
# overall rather than as a folded protein with disordered tails.
MOSTLY_DISORDERED_FRACTION = 0.5

_POSITIVE = {"LYS", "ARG"}
_NEGATIVE = {"ASP", "GLU"}


@dataclass(frozen=True, slots=True)
class StructuralFeatures:
    """Structural description of one protein, from its AlphaFold model."""

    accession: str
    n_residues: int
    mean_plddt: float
    fraction_disordered_plddt: float
    fraction_confident_plddt: float
    radius_of_gyration: float
    compactness: float
    relative_contact_order: float
    fraction_buried: float
    surface_net_charge: float
    buried_net_charge: float
    surface_charge_density: float

    @property
    def is_largely_disordered(self) -> bool:
        """Whether most of the chain is below AlphaFold's very-low-confidence band."""
        return self.fraction_disordered_plddt > MOSTLY_DISORDERED_FRACTION

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "accession": self.accession,
            "n_residues": self.n_residues,
            "mean_plddt": round(self.mean_plddt, 2),
            "fraction_disordered_plddt": round(self.fraction_disordered_plddt, 4),
            "fraction_confident_plddt": round(self.fraction_confident_plddt, 4),
            "radius_of_gyration": round(self.radius_of_gyration, 2),
            "compactness": round(self.compactness, 4),
            "relative_contact_order": round(self.relative_contact_order, 4),
            "fraction_buried": round(self.fraction_buried, 4),
            "surface_net_charge": round(self.surface_net_charge, 2),
            "buried_net_charge": round(self.buried_net_charge, 2),
            "surface_charge_density": round(self.surface_charge_density, 5),
            "is_largely_disordered": self.is_largely_disordered,
        }


def _coordinates(residues: Sequence[ResidueRecord]) -> np.ndarray:
    """Alpha-carbon coordinates for residues that have them."""
    return np.array(
        [residue.ca_coord for residue in residues if residue.ca_coord is not None],
        dtype=np.float64,
    )


def expected_radius_of_gyration(n_residues: int) -> float:
    """Radius of gyration expected for a folded globular protein of this length.

    The empirical scaling for compact native proteins, ``Rg = 2.2 * N**0.38``
    (Flory exponent near 3/5 for a self-avoiding walk, near 1/3 for a compact globule;
    the measured value for folded proteins sits between). Used only as a denominator, so
    a protein is called extended relative to what a globule of its own length would be -
    not relative to an absolute size, which would just re-measure length.
    """
    return 2.2 * (max(1, n_residues) ** 0.38)


def radius_of_gyration(coords: np.ndarray) -> float:
    """Root-mean-square distance of alpha carbons from their centroid."""
    if coords.size == 0:
        return 0.0
    centre = coords.mean(axis=0)
    return float(np.sqrt(((coords - centre) ** 2).sum(axis=1).mean()))


def neighbour_counts(coords: np.ndarray, *, radius: float = NEIGHBOUR_RADIUS) -> np.ndarray:
    """Number of other alpha carbons within ``radius`` of each residue."""
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    # Full pairwise distances. Proteins here are a few hundred residues, so the N^2 matrix
    # is small; chunking would cost more in complexity than it saves.
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    within = (distances <= radius).sum(axis=1) - 1  # exclude self
    return within.astype(np.int64)


def burial_from_neighbours(counts: np.ndarray, *, threshold: int = BURIED_NEIGHBOURS) -> np.ndarray:
    """Boolean burial per residue, from neighbour density.

    Isolated so a true solvent-accessible-surface calculation can replace it without
    changing any caller.
    """
    return counts >= threshold


def relative_contact_order(coords: np.ndarray) -> float:
    """Mean sequence separation of contacting residue pairs, divided by chain length.

    Zero would mean every contact is between sequence neighbours; high values mean the
    fold brings distant parts of the chain together. Contacts closer than
    :data:`MIN_CONTACT_SEPARATION` in sequence are excluded because backbone neighbours
    are always in contact and carry no information about topology.
    """
    n = coords.shape[0]
    if n <= MIN_CONTACT_SEPARATION:
        return 0.0

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    indices = np.arange(n)
    separation = np.abs(indices[:, None] - indices[None, :])

    contacts = (distances <= CONTACT_DISTANCE) & (separation >= MIN_CONTACT_SEPARATION)
    # Upper triangle only, so each pair is counted once.
    contacts = np.triu(contacts)
    n_contacts = int(contacts.sum())
    if n_contacts == 0:
        return 0.0
    return float(separation[contacts].sum() / (n_contacts * n))


def compute_structural_features(accession: str, residues: Sequence[ResidueRecord]) -> StructuralFeatures | None:
    """Derive every structural feature for one modelled protein.

    Returns ``None`` when the model has no usable alpha-carbon coordinates, rather than
    returning zeros that would be indistinguishable from a genuinely featureless protein.
    """
    coords = _coordinates(residues)
    if coords.shape[0] == 0:
        return None

    with_coords = [residue for residue in residues if residue.ca_coord is not None]
    plddts = [residue.plddt for residue in with_coords if residue.plddt is not None]

    n = coords.shape[0]
    counts = neighbour_counts(coords)
    buried = burial_from_neighbours(counts)

    charges = np.array(
        [
            1.0 if residue.resname in _POSITIVE else -1.0 if residue.resname in _NEGATIVE else 0.0
            for residue in with_coords
        ],
        dtype=np.float64,
    )
    surface_mask = ~buried
    n_surface = int(surface_mask.sum())

    observed_rg = radius_of_gyration(coords)
    expected_rg = expected_radius_of_gyration(n)

    return StructuralFeatures(
        accession=accession,
        n_residues=n,
        mean_plddt=float(np.mean(plddts)) if plddts else 0.0,
        fraction_disordered_plddt=(sum(1 for value in plddts if value < DISORDER_PLDDT) / len(plddts))
        if plddts
        else 0.0,
        fraction_confident_plddt=(sum(1 for value in plddts if value >= CONFIDENT_PLDDT) / len(plddts))
        if plddts
        else 0.0,
        radius_of_gyration=observed_rg,
        # Above 1.0 the chain is more extended than a globule of its length would be.
        compactness=observed_rg / expected_rg if expected_rg > 0 else 0.0,
        relative_contact_order=relative_contact_order(coords),
        fraction_buried=float(buried.mean()),
        surface_net_charge=float(charges[surface_mask].sum()),
        buried_net_charge=float(charges[buried].sum()),
        surface_charge_density=float(charges[surface_mask].sum() / n_surface) if n_surface else 0.0,
    )
