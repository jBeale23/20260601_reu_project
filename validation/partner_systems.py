"""What J-domain proteins partner when they are not partnering an Hsp70.

The pairing sweep asks which Hsp70 a co-chaperone works with, and deliberately drops
partners that are not Hsp70s from those axes - actin carries no "which Hsp70" label, and a
catch-all bucket would hold 99.4% of partners and dominate every table. ``PARTNER_SCOPE``
keeps the fact that such partners exist, but only as a single ``non_hsp70`` bit.

That bit is coarse. Of 8,638 distinct BioGRID partner genes in this set, 8,590 are not
Hsp70s, and they are not interchangeable: a J-domain protein whose non-chaperone partners
are ribosomal subunits is doing something different from one whose partners are proteasome
lids or cytoskeleton. This module asks whether class predicts *which system* a protein
touches outside the Hsp70 machinery.

Why prefixes rather than an ontology download
---------------------------------------------
The systems below are matched on gene-symbol prefixes, the same mechanism the Hsp70 axes
use. That is cruder than GO, but it is deliberate here: the GO route is exactly the
circularity the function work already ran into - unreviewed entries carry electronically
inferred terms, most often assigned from the domain signature by InterPro2GO, so testing
whether architecture predicts function through them assumes the answer. Symbol prefixes are
assigned by nomenclature committees from experimental identity, not from our features.

The cost is coverage: a symbol matching no system is reported as ``unassigned`` and counted
rather than forced into a bucket. That count is part of the result - a split that only
appears once most partners have been discarded is not a split.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.pairing import is_hsp70
from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Functional systems a J-domain protein can touch outside the Hsp70 machinery. Ordered by
# specificity: the first match wins, so narrower families are listed before broader ones.
#
# These are the systems chaperone biology actually distinguishes. Nucleotide exchange
# factors are separated from Hsp70 itself because they act *on* Hsp70 rather than being it;
# Hsp90 and the chaperonins are independent folding machines a JDP may hand off to; the
# ribosome and the proteasome mark the two ends of a nascent chain's life.
PARTNER_SYSTEMS: dict[str, tuple[str, ...]] = {
    "nucleotide_exchange_factor": ("BAG", "HSPBP", "FES1", "SIL1", "SSE", "GRPE", "HSPH", "HSP110"),
    "hsp90": ("HSP90", "HSPC", "HTPG", "HSC82", "HSP82", "TRAP1"),
    "chaperonin": ("CCT", "TCP1", "GROEL", "GROES", "HSPD", "HSPE", "MM_CPN"),
    "small_heat_shock": ("HSPB", "CRYAB", "HSP26", "HSP42", "IBPA", "IBPB"),
    "cochaperone_tpr": ("STI1", "HOP", "STIP", "CHIP", "STUB1", "CDC37", "AHA1"),
    "ribosome": ("RPL", "RPS", "MRPL", "MRPS", "RPP", "RPLP"),
    "translation_factor": ("EIF", "EEF", "TEF", "INFA", "INFB", "TUF"),
    "proteasome": ("PSM", "RPN", "RPT", "PRE", "PUP", "SCL1"),
    "ubiquitin_ligase": ("UBE", "UBC", "UBA", "RNF", "CUL", "SKP", "HUWE", "UBR", "DOA"),
    "cytoskeleton": ("ACT", "TUB", "MYO", "KRT", "VIM", "DYN", "KIF"),
    "protein_translocation": ("SEC", "SRP", "TOM", "TIM", "MAS", "OXA", "GET", "SND"),
    "vesicle_trafficking": ("VPS", "RAB", "ARF", "SNX", "COP", "CLTC", "AP2"),
    "transcription": ("POL", "TAF", "MED", "GTF", "TBP", "SPT", "SWI", "SNF"),
    "rna_processing": ("SNR", "SRSF", "HNRNP", "PRP", "LSM", "SM"),
    "metabolic_enzyme": ("PGK", "ENO", "GAPDH", "TDH", "PDC", "ADH", "FBA", "TPI"),
}

# A system needs this many proteins carrying it before a rate is worth quoting.
MIN_PROTEINS_PER_SYSTEM = 5

UNASSIGNED = "unassigned"

# Conventional false-discovery threshold, after Benjamini-Hochberg correction.
SIGNIFICANCE_LEVEL = 0.05


def classify_non_hsp70(gene: str) -> str | None:
    """Which functional system a non-Hsp70 partner belongs to.

    Returns ``None`` for an Hsp70 - those are the pairing sweep's business, not this
    module's - and :data:`UNASSIGNED` for a symbol matching no known system.
    """
    upper = gene.upper()
    if is_hsp70(upper):
        return None
    for system, prefixes in PARTNER_SYSTEMS.items():
        if any(upper.startswith(prefix) for prefix in prefixes):
            return system
    return UNASSIGNED


def systems_for(genes: Sequence[str]) -> set[str]:
    """Every non-Hsp70 system a protein's partners span."""
    found = {classify_non_hsp70(gene) for gene in genes}
    return {system for system in found if system is not None and system != UNASSIGNED}


@dataclass(frozen=True, slots=True)
class SystemAssociation:
    """Whether one class devotes more of its interactome to one system than the rest do.

    Composition, not presence. The obvious test - does this class touch system X at all -
    is confounded beyond use here, because median interactome size runs from 46 partners
    (``c_j_domain_only``) to 679 (``b_canonical``), a fifteen-fold spread. A protein with
    679 partners touches every system and one with 46 touches few, so presence/absence
    measures how well studied a protein is, not what it does. The fraction of a protein's
    *assigned* non-Hsp70 partners falling in a system is invariant to that.
    """

    system: str
    protein_class: str
    n_in_class: int
    n_in_others: int
    median_in_class: float
    median_in_others: float
    effect_size: float
    p_value: float
    p_adjusted: float = 1.0

    @property
    def direction(self) -> str:
        """Whether the class devotes more or less of its interactome to this system."""
        if self.median_in_class > self.median_in_others:
            return "enriched"
        return "depleted" if self.median_in_class < self.median_in_others else "equal"

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "system": self.system,
            "class": self.protein_class,
            "n_in_class": self.n_in_class,
            "n_in_others": self.n_in_others,
            "median_fraction_in_class": round(self.median_in_class, 5),
            "median_fraction_in_others": round(self.median_in_others, 5),
            "cliffs_delta": round(self.effect_size, 4),
            "direction": self.direction,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
        }


def system_fractions(genes: Sequence[str]) -> dict[str, float]:
    """Share of a protein's *assigned* non-Hsp70 partners in each system.

    Normalised over assigned partners rather than all of them, because 93.7% of partner
    mentions match no system in the prefix table; dividing by the total would make every
    fraction a measure of that coverage gap instead of of composition.
    """
    assigned: Counter[str] = Counter()
    for gene in genes:
        system = classify_non_hsp70(gene)
        if system is not None and system != UNASSIGNED:
            assigned[system] += 1
    total = sum(assigned.values())
    if total == 0:
        return {}
    return {system: count / total for system, count in assigned.items()}


def _cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    """Non-parametric effect size: P(x > y) - P(x < y) over the two samples."""
    if not first or not second:
        return 0.0
    greater = sum(1 for x in first for y in second if x > y)
    less = sum(1 for x in first for y in second if x < y)
    return (greater - less) / (len(first) * len(second))


def _mann_whitney_p(first: Sequence[float], second: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value by normal approximation with tie correction."""
    n1, n2 = len(first), len(second)
    if n1 == 0 or n2 == 0:
        return 1.0
    combined = sorted([(value, 0) for value in first] + [(value, 1) for value in second])
    ranks: list[float] = [0.0] * len(combined)
    tie_term = 0.0
    index = 0
    while index < len(combined):
        stop = index
        while stop + 1 < len(combined) and combined[stop + 1][0] == combined[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        group = stop - index + 1
        tie_term += group**3 - group
        index = stop + 1

    rank_sum = sum(rank for rank, (_, label) in zip(ranks, combined, strict=True) if label == 0)
    u_statistic = rank_sum - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    total = n1 + n2
    variance = (n1 * n2 / 12.0) * ((total + 1) - tie_term / (total * (total - 1))) if total > 1 else 0.0
    if variance <= 0:
        return 1.0
    z = (u_statistic - mean) / math.sqrt(variance)
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


def system_associations(
    classes: Mapping[str, str],
    interactions: Mapping[str, Sequence[str]],
) -> list[SystemAssociation]:
    """Test every class against every system on interactome composition."""
    shared = sorted(set(classes) & set(interactions))
    if not shared:
        return []

    fractions = {accession: system_fractions(interactions[accession]) for accession in shared}
    # Proteins with no assigned partner at all carry no composition and would otherwise
    # enter every comparison as a run of zeros.
    scored = [accession for accession in shared if fractions[accession]]
    by_class: dict[str, list[str]] = {}
    for accession in scored:
        by_class.setdefault(classes[accession], []).append(accession)

    systems = sorted({system for values in fractions.values() for system in values})
    results: list[SystemAssociation] = []
    for system in systems:
        carriers = sum(1 for accession in scored if fractions[accession].get(system))
        if carriers < MIN_PROTEINS_PER_SYSTEM:
            continue
        for protein_class, members in by_class.items():
            in_class = [fractions[accession].get(system, 0.0) for accession in members]
            in_others = [
                fractions[accession].get(system, 0.0) for accession in scored if classes[accession] != protein_class
            ]
            if not in_class or not in_others:
                continue
            results.append(
                SystemAssociation(
                    system=system,
                    protein_class=protein_class,
                    n_in_class=len(in_class),
                    n_in_others=len(in_others),
                    median_in_class=statistics.median(in_class),
                    median_in_others=statistics.median(in_others),
                    effect_size=_cliffs_delta(in_class, in_others),
                    p_value=_mann_whitney_p(in_class, in_others),
                ),
            )

    adjusted = benjamini_hochberg({f"{item.system}|{item.protein_class}": item.p_value for item in results})
    return [
        SystemAssociation(
            system=item.system,
            protein_class=item.protein_class,
            n_in_class=item.n_in_class,
            n_in_others=item.n_in_others,
            median_in_class=item.median_in_class,
            median_in_others=item.median_in_others,
            effect_size=item.effect_size,
            p_value=item.p_value,
            p_adjusted=adjusted[f"{item.system}|{item.protein_class}"],
        )
        for item in results
    ]


def non_hsp70_report(
    classes: Mapping[str, str],
    interactions: Mapping[str, Sequence[str]],
    *,
    max_reported: int = 40,
) -> dict[str, object]:
    """The non-Hsp70 interactome, by class.

    Reports the unassigned share prominently. A class difference that only emerges after
    most partners fall outside every system is a statement about the prefix table, not
    about biology.
    """
    shared = sorted(set(classes) & set(interactions))
    partner_counts: Counter[str] = Counter()
    unassigned_symbols: Counter[str] = Counter()
    for accession in shared:
        for gene in interactions[accession]:
            system = classify_non_hsp70(gene)
            if system is None:
                continue
            partner_counts[system] += 1
            if system == UNASSIGNED:
                unassigned_symbols[gene.upper()] += 1

    # Interactome size per class, reported first because it is the confound that ruins the
    # obvious version of this analysis and a reader must be able to see it.
    sizes_by_class: dict[str, list[int]] = {}
    for accession in shared:
        sizes_by_class.setdefault(classes[accession], []).append(len(interactions[accession]))
    interactome_sizes = {
        protein_class: {
            "n_proteins": len(sizes),
            "median_partners": statistics.median(sizes),
            "mean_partners": round(statistics.mean(sizes), 1),
        }
        for protein_class, sizes in sorted(sizes_by_class.items())
    }

    total_non_hsp70 = sum(partner_counts.values())
    associations = system_associations(classes, interactions)
    associations.sort(key=lambda item: (item.p_adjusted, -abs(item.effect_size)))

    return {
        "n_proteins_tested": len(shared),
        "interactome_size_by_class": interactome_sizes,
        "n_non_hsp70_partner_mentions": total_non_hsp70,
        "unassigned_fraction": round(partner_counts[UNASSIGNED] / total_non_hsp70, 4) if total_non_hsp70 else 0.0,
        "partners_by_system": dict(partner_counts.most_common()),
        "most_common_unassigned_symbols": dict(unassigned_symbols.most_common(20)),
        "n_associations_tested": len(associations),
        "n_significant": sum(1 for item in associations if item.p_adjusted < SIGNIFICANCE_LEVEL),
        "associations": [item.to_json_dict() for item in associations[:max_reported]],
        "note": (
            "Tests compare the *fraction* of a protein's assigned non-Hsp70 partners in "
            "each system, not whether it touches the system at all. Presence/absence is "
            "unusable here: median interactome size runs from 46 partners to 679 across "
            "classes, so touching a system measures how well studied a protein is. "
            "Systems are matched on gene-symbol prefixes rather than GO, deliberately: GO "
            "terms on unreviewed entries are electronically inferred from the domain "
            "signature by InterPro2GO, so using them to test whether architecture predicts "
            "partner choice would assume the answer. Symbols are assigned from "
            "experimental identity instead. The cost is coverage, which is why the "
            "unassigned fraction is reported beside every result."
        ),
    }
