"""Carrying disease association across orthologs.

Ten J-domain proteins in this set carry a curated disease. That is too few to test anything,
and the reason is not that only ten matter - it is that disease annotation is assigned to
the human protein and to almost nothing else. A mouse *Dnajb6* knockout with a myopathic
phenotype tells us about DNAJB6, but UniProt records the disease against the human entry
alone.

Transferring the association across orthologs recovers that. It multiplies the labelled set
by roughly the number of species represented, at the cost of an assumption: that orthologs
share the function whose loss causes the disease.

Where that assumption breaks
----------------------------
It breaks exactly where this project is most interested. A J-domain protein's function is
carried substantially by its non-J regions, and those are the least conserved part; two
orthologs can share a J-domain at 90% identity and share almost nothing else. So transfer
is gated on identity over the **non-J regions**, not over the whole sequence, and the
threshold is deliberately high.

The transferred label is also marked as transferred, everywhere, and never merged silently
into the observed set. An analysis is free to use both, but it has to say which it used -
because a result that appears only after transfer is a result about the transfer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.clustering import sequence_identity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Identity over non-J regions required to carry a disease association across. High on
# purpose: the claim being transferred is that losing this protein causes this disease, and
# a distant relative with a different C-terminus is a different protein functionally.
DEFAULT_TRANSFER_IDENTITY = 0.60

# Below this, a non-J region is too short for its identity to mean anything.
MIN_REGION_LENGTH = 30


@dataclass(frozen=True, slots=True)
class TransferredDisease:
    """A disease association carried from one protein to a relative."""

    accession: str
    donor: str
    diseases: tuple[str, ...]
    identity: float
    donor_organism: str = ""
    organism: str = ""

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "accession": self.accession,
            "donor": self.donor,
            "diseases": list(self.diseases),
            "identity": round(self.identity, 4),
            "donor_organism": self.donor_organism,
            "organism": self.organism,
            "evidence": "transferred",
        }


def transfer_disease(
    donors: Mapping[str, Sequence[str]],
    donor_regions: Mapping[str, str],
    target_regions: Mapping[str, str],
    *,
    identity_threshold: float = DEFAULT_TRANSFER_IDENTITY,
    organisms: Mapping[str, str] | None = None,
) -> dict[str, TransferredDisease]:
    """Carry disease associations from annotated proteins to their close relatives.

    Args:
        donors: Disease names per annotated accession.
        donor_regions: Non-J sequence per donor, which is what identity is measured over.
        target_regions: Non-J sequence per candidate recipient.
        identity_threshold: Minimum identity to carry an association.
        organisms: Optional organism names, recorded so a transfer can be audited.

    Returns:
        The best transfer per target - one donor, the closest that clears the threshold.
    """
    names = organisms or {}
    usable_donors = {
        accession: region
        for accession, region in donor_regions.items()
        if accession in donors and len(region) >= MIN_REGION_LENGTH
    }
    if not usable_donors:
        logger.warning("no donors with a non-J region long enough to transfer from")
        return {}

    transferred: dict[str, TransferredDisease] = {}
    for accession, region in target_regions.items():
        if accession in donors or len(region) < MIN_REGION_LENGTH:
            continue
        best: tuple[float, str] | None = None
        for donor, donor_region in usable_donors.items():
            identity = sequence_identity(region, donor_region)
            if identity >= identity_threshold and (best is None or identity > best[0]):
                best = (identity, donor)
        if best is not None:
            identity, donor = best
            transferred[accession] = TransferredDisease(
                accession=accession,
                donor=donor,
                diseases=tuple(donors[donor]),
                identity=identity,
                donor_organism=names.get(donor, ""),
                organism=names.get(accession, ""),
            )
    return transferred


def transfer_report(
    observed: Mapping[str, Sequence[str]],
    transferred: Mapping[str, TransferredDisease],
    *,
    identity_threshold: float = DEFAULT_TRANSFER_IDENTITY,
    max_reported: int = 40,
) -> dict[str, object]:
    """What transfer added, kept separate from what was observed."""
    donors_used = {item.donor for item in transferred.values()}
    species = {item.organism for item in transferred.values() if item.organism}
    return {
        "n_observed": len(observed),
        "n_transferred": len(transferred),
        "n_donors_used": len(donors_used),
        "n_species_reached": len(species),
        "identity_threshold": identity_threshold,
        "transfers": [item.to_json_dict() for item in list(transferred.values())[:max_reported]],
        "note": (
            "Identity is measured over non-J regions, not whole sequences. A J-domain "
            "protein's function is carried substantially by its non-J regions, and those "
            "are the least conserved part - two orthologs can share a J-domain at 90% "
            "identity and share almost nothing else, so whole-sequence identity would "
            "carry disease associations across proteins that do different jobs. Every "
            "transferred label is marked as transferred and never merged into the observed "
            "set: a result that appears only after transfer is a result about the transfer."
        ),
    }
