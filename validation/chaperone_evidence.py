"""Is this protein plausibly a chaperone, or only a J-domain match?

A real concern about widening the dataset: the top twenty architectures are unambiguously
Hsp40 co-chaperones, but the long tail might contain proteins that merely match the
J-domain signature without functioning as chaperones. Every downstream claim about class
structure would then be contaminated by proteins that do not belong to the family at all.

The sharpest available test is the **HPD motif** - histidine, proline, aspartate - in helix
II of the J-domain. It is what physically contacts Hsp70 and triggers ATP hydrolysis, and
mutating it abolishes co-chaperone activity while leaving the fold intact. A protein with a
J-domain but no HPD has a J-*like* domain: same shape, no chaperone function.

Measured on the full set, HPD presence does not decay with architecture rarity - 96.9% in
the five commonest architectures, 98.3% in ranks 21-50, 95.7% beyond rank 50. The tail is
not less chaperone-like than the head, which is the result this module exists to keep
checking as the dataset grows.

What this does not settle
-------------------------
HPD presence is necessary, not sufficient. A protein can carry an intact J-domain and still
act mainly through another domain - the Myb/SANT DNA-binding domains found among the rarer
architectures are a real example, where the protein may be a transcriptional regulator that
happens to recruit Hsp70. Those are flagged separately rather than excluded, because
whether they belong is a judgement about scope rather than a fact about the sequence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain_layout.pipeline import ProteinLayout

# Domain families whose presence suggests the protein's principal activity may lie
# elsewhere, even with an intact J-domain. Not grounds for exclusion - a co-chaperone can
# legitimately carry a DNA-binding domain - but grounds for reporting separately.
NON_CHAPERONE_HINTS = frozenset(
    {
        "myb_like",
        "sant",
        "dna_binding",
        "homeobox",
        "zinc_finger_c2h2",
        "kinase",
    }
)

# HPD rate below which an architecture is worth inspecting rather than trusting. Set well
# under the 95.7% observed in the rarest architectures, so it fires on a genuine departure
# rather than on ordinary annotation noise.
SUSPECT_HPD_RATE = 0.80

# Architectures with fewer members than this are not judged on their HPD rate: with three
# proteins, one missing motif looks like a 33% failure.
MIN_MEMBERS_TO_JUDGE = 20


@dataclass(frozen=True, slots=True)
class ArchitectureEvidence:
    """Chaperone plausibility for one domain architecture."""

    architecture: str
    n_proteins: int
    n_with_j_domain: int
    n_with_hpd: int
    n_with_non_chaperone_hint: int

    @property
    def hpd_rate(self) -> float:
        """Share carrying the motif that makes a J-domain functional."""
        return self.n_with_hpd / self.n_proteins if self.n_proteins else 0.0

    @property
    def is_judgeable(self) -> bool:
        """Whether this architecture has enough members for its rate to mean anything."""
        return self.n_proteins >= MIN_MEMBERS_TO_JUDGE

    @property
    def is_suspect(self) -> bool:
        """Whether the architecture departs from the family's normal HPD rate."""
        return self.is_judgeable and self.hpd_rate < SUSPECT_HPD_RATE

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "architecture": self.architecture,
            "n_proteins": self.n_proteins,
            "n_with_j_domain": self.n_with_j_domain,
            "n_with_hpd": self.n_with_hpd,
            "hpd_rate": round(self.hpd_rate, 4),
            "n_with_non_chaperone_hint": self.n_with_non_chaperone_hint,
            "suspect": self.is_suspect,
        }


def architecture_evidence(layouts: Sequence[ProteinLayout]) -> dict[str, ArchitectureEvidence]:
    """Chaperone plausibility per architecture, over a whole run."""
    grouped: dict[str, list[ProteinLayout]] = {}
    for layout in layouts:
        key = layout.evidence.domain_family_layout or "(no annotated domains)"
        grouped.setdefault(key, []).append(layout)

    evidence: dict[str, ArchitectureEvidence] = {}
    for architecture, members in grouped.items():
        hints = sum(
            1 for item in members if any(family in NON_CHAPERONE_HINTS for family in item.evidence.structured_families)
        )
        evidence[architecture] = ArchitectureEvidence(
            architecture=architecture,
            n_proteins=len(members),
            n_with_j_domain=sum(1 for item in members if item.evidence.has_j_domain),
            n_with_hpd=sum(1 for item in members if item.evidence.has_hpd),
            n_with_non_chaperone_hint=hints,
        )
    return evidence


def chaperone_report(layouts: Sequence[ProteinLayout], *, max_suspect: int = 25) -> dict[str, object]:
    """Whether widening the dataset admitted proteins that are not chaperones.

    Reports the HPD rate banded by how common an architecture is, because the concern is
    specifically that the rare tail differs from the head. If the bands agree, the tail is
    not less chaperone-like than the head and the widening is safe.
    """
    evidence = architecture_evidence(layouts)
    if not evidence:
        return {"n_architectures": 0, "note": "no layouts to assess"}

    ranked = sorted(evidence.values(), key=lambda item: -item.n_proteins)
    bands: dict[str, object] = {}
    for label, low, high in (
        ("rank_1_5", 1, 5),
        ("rank_6_20", 6, 20),
        ("rank_21_50", 21, 50),
        ("rank_51_plus", 51, None),
    ):
        selected = ranked[low - 1 : high]
        total = sum(item.n_proteins for item in selected)
        with_hpd = sum(item.n_with_hpd for item in selected)
        if total:
            bands[label] = {
                "n_architectures": len(selected),
                "n_proteins": total,
                "hpd_rate": round(with_hpd / total, 4),
            }

    suspect = [item for item in ranked if item.is_suspect]
    hinted = [item for item in ranked if item.n_with_non_chaperone_hint > 0]

    return {
        "n_architectures": len(evidence),
        "hpd_rate_by_architecture_rank": bands,
        "n_suspect_architectures": len(suspect),
        "suspect_architectures": [item.to_json_dict() for item in suspect[:max_suspect]],
        "n_architectures_with_non_chaperone_hint": len(hinted),
        "interpretation": (
            "HPD is the motif that makes a J-domain functional, so a rate that holds across "
            "architecture ranks means the rare tail is no less chaperone-like than the head. "
            "A rate that falls off with rarity would mean widening the dataset admitted "
            "proteins that merely match the signature. HPD presence is necessary but not "
            "sufficient: a protein with an intact J-domain and a DNA-binding domain may act "
            "principally as a transcriptional regulator, and those are counted separately "
            "rather than excluded."
        ),
    }


def summarize_hints(layouts: Sequence[ProteinLayout], *, top: int = 12) -> list[tuple[str, int]]:
    """Which non-chaperone-suggesting families appear, and how often."""
    counts: Counter[str] = Counter()
    for layout in layouts:
        for family in layout.evidence.structured_families:
            if family in NON_CHAPERONE_HINTS:
                counts[family] += 1
    return counts.most_common(top)
