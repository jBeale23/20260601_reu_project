"""Does a J-domain protein's class predict which Hsp70 it partners?

This is the functional claim the project has been building toward. A reclassification that
only reorganises annotation is bookkeeping; one that predicts a protein's partner is a
statement about biology. J-domain proteins act *on* Hsp70s, so partner identity is the most
direct functional property they have.

Three ways to describe a partner, and every combination is tested
-----------------------------------------------------------------
There is no single right granularity, and the three available answer different questions:

**Compartment** - cytosolic against endoplasmic-reticulum and mitochondrial Hsp70s. Coarse,
well populated, and driven by localisation. The most likely to show signal and the least
specific about mechanism.

**Functional subtype** - SSA-type (general folding) against SSB-type (ribosome-associated,
acting on nascent chains). Sharper biologically: zuotin partners SSB alone, which is a real
specificity that compartment cannot express.

**Paralogue** - HSPA1A against HSPA8 and so on. Finest, and the most fragile, because
paralogues are near-identical and interaction databases frequently cannot tell them apart.

Running one and reporting it would invite the reader to assume the others agree. All seven
non-empty combinations are run, so a result that holds only at one granularity is visible
as such - and if compartment separates the classes while paralogue does not, that is itself
the finding: the signal is about where a chaperone works, not which copy of it.

Two evidence sources, and every combination of those too
---------------------------------------------------------
**Interaction** evidence from BioGRID is direct and experimental, and exists almost only
for model organisms. **Co-occurrence** asks whether a JDP subclass appears only in genomes
that carry a particular Hsp70 subtype; it is correlational but reaches the whole set. Used
alone each is misleading in a different direction, so interaction-only, co-occurrence-only,
and combined are all reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from scipy.stats import chi2_contingency, fisher_exact

from validation.recurrence import benjamini_hochberg, genus_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Granularity axes. Every non-empty combination of these is tested.
COMPARTMENT = "compartment"
SUBTYPE = "functional_subtype"
PARALOGUE = "paralogue"

# Whether a partner is an Hsp70 at all.
#
# The other three axes describe *which* Hsp70 a protein partners, so a partner that is not
# an Hsp70 - actin, a transcription factor - has no label on them and drops out. That is
# right for those axes and wrong as a whole: how much of a J-domain protein's interactome
# is Hsp70 is itself a functional property. A co-chaperone whose partners are mostly Hsp70
# is doing chaperone work; one with two Hsp70 partners among two hundred others is
# probably doing something else and recruiting Hsp70 incidentally. This axis makes that
# visible and testable instead of discarding it.
PARTNER_SCOPE = "partner_scope"

GRANULARITIES = (COMPARTMENT, SUBTYPE, PARALOGUE, PARTNER_SCOPE)

# Evidence sources, likewise combined exhaustively.
INTERACTION = "interaction"

# Co-occurrence at two taxonomic units, because the right one is not obvious.
#
# Genus matches more J-domain proteins - many come from organisms with no sequenced Hsp70
# in the set - but two species in a genus can differ in Hsp70 complement, so a genus match
# may pair a protein with a chaperone its own species does not have. Species is stricter
# and shrinks the matched set sharply. Reporting both makes the choice visible: if the
# association holds at species as well as genus, the coarser unit was not manufacturing it.
CO_OCCURRENCE_GENUS = "co_occurrence_genus"
CO_OCCURRENCE_SPECIES = "co_occurrence_species"

EVIDENCE_SOURCES = (INTERACTION, CO_OCCURRENCE_GENUS, CO_OCCURRENCE_SPECIES)

SIGNIFICANCE_FDR = 0.05

# Pairs required before a contingency table is tested. Below this the chi-square
# approximation fails and Fisher's exact test carries no power worth reporting.
MIN_PAIRS = 20

# A contingency table needs at least two rows and two columns to express an association.
MIN_TABLE_DIMENSION = 2

# Cramer's V thresholds, the conventional bands.
LARGE_EFFECT = 0.5
MEDIUM_EFFECT = 0.3
SMALL_EFFECT = 0.1

# Gene-name prefixes mapping an Hsp70 to a compartment. Localisation is the property these
# express most reliably across organisms, which is why it is the coarsest axis.
_COMPARTMENT_PREFIXES = {
    "cytosol": ("SSA", "SSB", "HSPA1", "HSPA2", "HSPA6", "HSPA8", "HSPA14", "HSC70", "HSP70"),
    "endoplasmic_reticulum": ("KAR2", "BIP", "HSPA5", "GRP78", "LHS1"),
    "mitochondrion": ("SSC", "SSQ", "HSPA9", "MORTALIN", "MTHSP", "ECM10"),
    "chloroplast": ("CSS1", "HSP70B", "CGE1"),
    "bacterial": ("DNAK", "HSCA", "HSCB"),
}

# Functional subtype. SSB is ribosome-associated and acts on nascent chains; SSA is the
# general folding machine. The distinction is what makes zuotin's specificity meaningful.
_SUBTYPE_PREFIXES = {
    "ssb_ribosome_associated": ("SSB",),
    "ssa_general_folding": ("SSA", "HSPA1", "HSPA8", "HSC70"),
    "bip_er_lumenal": ("KAR2", "BIP", "HSPA5", "GRP78"),
    "mitochondrial_import": ("SSC", "HSPA9", "MORTALIN"),
    "bacterial_dnak": ("DNAK",),
}


def classify_partner(gene: str, granularity: str) -> str | None:
    """Label one Hsp70 partner at the requested granularity.

    Returns ``None`` when the gene cannot be placed, so an unclassifiable partner is
    dropped from that comparison rather than silently joining a catch-all bucket that would
    then dominate every table.
    """
    upper = canonical_paralogue(gene)
    if upper in _NOT_HSP70:
        return None
    if granularity == PARALOGUE:
        # Hsp70-only, for the same reason compartment and subtype are: this axis asks
        # *which Hsp70 paralogue* a protein partners. Returning the bare symbol for
        # anything at all made every non-Hsp70 partner its own class - on the real BioGRID
        # data, 8,364 classes where there are 48 Hsp70s - which inflates the contingency
        # table's degrees of freedom, dilutes Cramer's V toward zero, and quietly changes
        # the question to "which of any protein". Non-Hsp70 partners are not lost: they
        # are what PARTNER_SCOPE and hsp70_partner_fraction measure.
        return upper if is_hsp70(upper) else None
    if granularity == PARTNER_SCOPE:
        return "hsp70" if is_hsp70(upper) else "non_hsp70"
    table = _COMPARTMENT_PREFIXES if granularity == COMPARTMENT else _SUBTYPE_PREFIXES
    for label, prefixes in table.items():
        if any(upper.startswith(prefix) for prefix in prefixes):
            return label
    return None


# Gene symbols that a prefix match wrongly claims for Hsp70.
#
# Prefix matching is cheap and mostly right, and these are the cases where it is badly
# wrong - each is an abundant, promiscuous interactor whose symbol merely begins the same
# way, so counting it inflates exactly the association being measured:
#
#   SSB    in bacteria is single-stranded DNA-binding protein, among the most common
#          interactors in E. coli datasets - not yeast Ssb1/Ssb2, the ribosome-associated
#          Hsp70 the prefix is meant to catch. It accounted for 660 of the partner calls.
#   SSBP*  single-stranded DNA-binding protein, likewise unrelated.
#   BIPA   a bacterial GTPase (TypA). BiP itself is HSPA5 or KAR2, both matched separately.
#
# Matched on the exact symbol rather than a prefix, because the real Hsp70s SSB1 and SSB2
# must still be caught.
_NOT_HSP70 = frozenset({"SSB", "SSBP", "SSBP1", "SSBP2", "SSBP3", "SSBP4", "BIPA"})

# One Hsp70 under its many names. Gene symbols for the same chaperone differ by organism and
# by era - the ER's BiP is HSPA5 in human, KAR2 in yeast, and appears as BIP, BIP1, BIP3 and
# GRP78 across the interaction databases - and treating those as distinct partners splits one
# answer into six. That is not cosmetic: a correct prediction of HSPA5 scores as an error
# whenever the recorded label happens to read BIP, so fragmentation depresses every accuracy
# and every association measured on this readout.
#
# Only unambiguous synonyms are merged, and only within a compartment. The generic labels -
# HSP70, HSPA - are deliberately absent: they name no particular paralogue, and folding them
# into one would assert a specificity the data does not have.
_PARALOGUE_SYNONYMS = {
    # BiP: the ER Hsp70.
    "BIP": "HSPA5",
    "BIP1": "HSPA5",
    "BIP2": "HSPA5",
    "BIP3": "HSPA5",
    "GRP78": "HSPA5",
    "KAR2": "HSPA5",
    # Hsc70: the constitutive cytosolic Hsp70.
    "HSC70": "HSPA8",
    "HSP73": "HSPA8",
    "HSPA10": "HSPA8",
    # Mortalin: the mitochondrial Hsp70.
    "GRP75": "HSPA9",
    "MOT2": "HSPA9",
    "SSC1": "HSPA9",
    "MTHSP70": "HSPA9",
    # Hsp70-1: the stress-inducible cytosolic pair, which are near-identical proteins.
    "HSPA1B": "HSPA1A",
    "HSP72": "HSPA1A",
}


def canonical_paralogue(gene: str) -> str:
    """One name per Hsp70 paralogue, folding known synonyms together.

    Returns the symbol unchanged when it is not a recognised synonym, so an unfamiliar or
    generic name is never silently reassigned to a paralogue it may not be.
    """
    upper = gene.upper()
    return _PARALOGUE_SYNONYMS.get(upper, upper)


def is_hsp70(gene: str) -> bool:
    """Whether a partner belongs to the Hsp70 family at all.

    Exclusions are checked first: a symbol on the not-Hsp70 list is refused even when it
    begins like one, because the alternative is counting an abundant DNA-binding protein
    as a chaperone in every test that rests on this function.
    """
    upper = gene.upper()
    if upper in _NOT_HSP70:
        return False
    # Canonicalise first. Several Hsp70s are named nothing like "HSPA" - yeast's BiP is KAR2,
    # its mitochondrial Hsp70 is SSC1, and the ER chaperone appears as GRP78 - so a
    # prefix test applied to the raw symbol does not merely split them, it fails to
    # recognise them as Hsp70s at all and drops them from the readout entirely.
    upper = canonical_paralogue(upper)
    if upper in _NOT_HSP70:
        return False
    return any(any(upper.startswith(prefix) for prefix in prefixes) for prefixes in _COMPARTMENT_PREFIXES.values())


def hsp70_partner_fraction(genes: Sequence[str]) -> float:
    """Share of a protein's partners that are Hsp70s.

    Reported alongside the pairing tests, because a class difference in *which* Hsp70 is
    partnered means something different when one class barely partners Hsp70s at all.
    """
    if not genes:
        return 0.0
    return sum(1 for gene in genes if is_hsp70(gene)) / len(genes)


def partner_labels(genes: Sequence[str], granularities: Sequence[str]) -> set[str]:
    """Every label a set of partners carries, across the requested axes.

    Combining axes concatenates them, so a protein partnering SSB1 under compartment plus
    subtype contributes both "cytosol" and "ssb_ribosome_associated" - which is what lets a
    combination detect a pattern that neither axis resolves alone.
    """
    labels: set[str] = set()
    for gene in genes:
        for granularity in granularities:
            label = classify_partner(gene, granularity)
            if label is not None:
                labels.add(f"{granularity}:{label}")
    return labels


@dataclass(frozen=True, slots=True)
class PairingResult:
    """One test of whether JDP class predicts Hsp70 partner label."""

    granularities: tuple[str, ...]
    evidence: tuple[str, ...]
    n_proteins: int
    n_labels: int
    chi2: float
    p_value: float
    cramers_v: float
    p_adjusted: float = 1.0

    @property
    def is_significant(self) -> bool:
        """Whether the association survives multiple-testing correction."""
        return self.p_adjusted < SIGNIFICANCE_FDR

    @property
    def effect_size_label(self) -> str:
        """Cramer's V in words; the number alone invites over-reading a large table."""
        if self.cramers_v >= LARGE_EFFECT:
            return "large"
        if self.cramers_v >= MEDIUM_EFFECT:
            return "medium"
        if self.cramers_v >= SMALL_EFFECT:
            return "small"
        return "negligible"

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "granularities": list(self.granularities),
            "evidence": list(self.evidence),
            "n_proteins": self.n_proteins,
            "n_partner_labels": self.n_labels,
            "chi2": round(self.chi2, 3),
            "cramers_v": round(self.cramers_v, 4),
            "effect_size": self.effect_size_label,
            "p_value": self.p_value,
            "p_adjusted": round(self.p_adjusted, 6),
            "significant": self.is_significant,
        }


def _cramers_v(chi2: float, n: int, rows: int, columns: int) -> float:
    """Effect size for a contingency table, bounded in [0, 1].

    Chi-square grows with sample size, so on tens of thousands of proteins every table is
    significant; Cramer's V is what says whether the association is large.
    """
    smaller = min(rows - 1, columns - 1)
    if n <= 0 or smaller <= 0:
        return 0.0
    return float((chi2 / (n * smaller)) ** 0.5)


def test_pairing(
    classes: Mapping[str, str],
    partner_sets: Mapping[str, Sequence[str]],
    granularities: Sequence[str],
    evidence: Sequence[str],
) -> PairingResult | None:
    """Test one granularity/evidence combination for class-partner association."""
    table: dict[str, dict[str, int]] = {}
    labelled = 0
    for accession, jdp_class in classes.items():
        genes = partner_sets.get(accession)
        if not genes:
            continue
        labels = partner_labels(genes, granularities)
        if not labels:
            continue
        labelled += 1
        row = table.setdefault(jdp_class, {})
        for label in labels:
            row[label] = row.get(label, 0) + 1

    if labelled < MIN_PAIRS or len(table) < MIN_TABLE_DIMENSION:
        return None

    columns = sorted({label for row in table.values() for label in row})
    if len(columns) < MIN_TABLE_DIMENSION:
        return None
    matrix = [[row.get(label, 0) for label in columns] for row in table.values()]

    try:
        chi2, p_value, _dof, _expected = chi2_contingency(matrix)
    except ValueError:
        return None

    total = sum(sum(row) for row in matrix)
    return PairingResult(
        granularities=tuple(granularities),
        evidence=tuple(evidence),
        n_proteins=labelled,
        n_labels=len(columns),
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=_cramers_v(float(chi2), total, len(matrix), len(columns)),
    )


@dataclass(frozen=True, slots=True)
class EvidenceSets:
    """Partner sets from each source, kept apart so combinations can be assembled."""

    interaction: Mapping[str, Sequence[str]]
    co_occurrence_genus: Mapping[str, Sequence[str]]
    co_occurrence_species: Mapping[str, Sequence[str]]

    def by_source(self, source: str) -> Mapping[str, Sequence[str]]:
        """The partner set for one named source."""
        return {
            INTERACTION: self.interaction,
            CO_OCCURRENCE_GENUS: self.co_occurrence_genus,
            CO_OCCURRENCE_SPECIES: self.co_occurrence_species,
        }.get(source, {})


def _combined_partners(sets: EvidenceSets, sources: Sequence[str]) -> dict[str, list[str]]:
    """Partner sets from the requested evidence sources, merged and de-duplicated."""
    merged: dict[str, list[str]] = {}
    for source in sources:
        for accession, genes in sets.by_source(source).items():
            merged.setdefault(accession, []).extend(genes)
    return {accession: list(dict.fromkeys(genes)) for accession, genes in merged.items()}


def sweep(classes: Mapping[str, str], sets: EvidenceSets) -> dict[str, object]:
    """Every combination of granularity and evidence, corrected together.

    Fifteen granularity combinations times seven evidence combinations. Correcting across
    all of them is the point: running a hundred tests and reporting the best one
    uncorrected would manufacture a result, and the sweep exists precisely so that cannot
    happen silently. Both a raw p-value and a Benjamini-Hochberg adjusted one are reported
    for every combination.
    """
    granularity_sets = [
        list(subset) for size in range(1, len(GRANULARITIES) + 1) for subset in combinations(GRANULARITIES, size)
    ]
    evidence_sets = [
        list(subset) for size in range(1, len(EVIDENCE_SOURCES) + 1) for subset in combinations(EVIDENCE_SOURCES, size)
    ]

    results: list[PairingResult] = []
    for evidence in evidence_sets:
        partners = _combined_partners(sets, evidence)
        for granularities in granularity_sets:
            result = test_pairing(classes, partners, granularities, evidence)
            if result is not None:
                results.append(result)

    adjusted = benjamini_hochberg(
        {f"{'+'.join(r.granularities)}|{'+'.join(r.evidence)}": r.p_value for r in results},
    )
    corrected = [
        PairingResult(
            granularities=r.granularities,
            evidence=r.evidence,
            n_proteins=r.n_proteins,
            n_labels=r.n_labels,
            chi2=r.chi2,
            p_value=r.p_value,
            cramers_v=r.cramers_v,
            p_adjusted=adjusted[f"{'+'.join(r.granularities)}|{'+'.join(r.evidence)}"],
        )
        for r in results
    ]
    corrected.sort(key=lambda r: (-r.cramers_v, r.p_adjusted))

    significant = [r for r in corrected if r.is_significant]
    return {
        "n_combinations_tested": len(corrected),
        "n_significant": len(significant),
        "significance_fdr": SIGNIFICANCE_FDR,
        "results": [r.to_json_dict() for r in corrected],
        "hsp70_partner_fraction_by_class": _hsp70_fraction_by_class(classes, sets),
        "interpretation": (
            "Every combination is tested and corrected together, so a result that "
            "appears at one granularity and not others is visible as such rather than "
            "reported alone. Cramer's V rather than the p-value carries the meaning: on "
            "tens of thousands of proteins every table is significant, and what matters is "
            "whether the association is large. If compartment separates the classes while "
            "paralogue does not, the signal is about where a chaperone works rather than "
            "which copy of it."
        ),
    }


def _hsp70_fraction_by_class(classes: Mapping[str, str], sets: EvidenceSets) -> dict[str, object]:
    """Median share of each class's interaction partners that are Hsp70s.

    Context for every pairing result: a difference in *which* Hsp70 is partnered means
    something different when one class barely partners Hsp70s at all.
    """
    grouped: dict[str, list[float]] = {}
    for accession, jdp_class in classes.items():
        genes = sets.interaction.get(accession)
        if genes:
            grouped.setdefault(jdp_class, []).append(hsp70_partner_fraction(genes))
    return {
        jdp_class: {
            "n_with_partners": len(values),
            "median_hsp70_fraction": round(sorted(values)[len(values) // 2], 4),
        }
        for jdp_class, values in sorted(grouped.items())
    }


def co_occurrence_partners(
    jdp_organisms: Mapping[str, str],
    hsp70_by_genus: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Hsp70s present in the same genus as each J-domain protein.

    Correlational by construction - sharing a genome is not interacting - but it reaches
    the whole set rather than only the organisms somebody has assayed, which is where the
    interaction evidence stops.
    """
    return {
        accession: list(hsp70_by_genus.get(genus_of(organism), ())) for accession, organism in jdp_organisms.items()
    }


def species_of(organism_name: str) -> str:
    """First two tokens of a binomial name, used as the finer co-occurrence unit."""
    parts = (organism_name or "").strip().split()
    return " ".join(parts[:2]) if len(parts) >= MIN_TABLE_DIMENSION else (parts[0] if parts else "unknown")


def hsp70_index_by_species(
    hsp70_names: Mapping[str, str],
    hsp70_organisms: Mapping[str, str],
) -> dict[str, list[str]]:
    """Which Hsp70 gene names occur in each species."""
    index: dict[str, set[str]] = {}
    for accession, organism in hsp70_organisms.items():
        name = hsp70_names.get(accession)
        if name:
            index.setdefault(species_of(organism), set()).add(name)
    return {species: sorted(names) for species, names in index.items()}


def co_occurrence_partners_by_species(
    jdp_organisms: Mapping[str, str],
    hsp70_by_species: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Hsp70s present in the same species as each J-domain protein.

    Stricter than the genus pairing and matches far fewer proteins, since many organisms
    here have no sequenced Hsp70 in the set. Reported beside the genus result so the
    coarser unit cannot manufacture an association unnoticed.
    """
    return {
        accession: list(hsp70_by_species.get(species_of(organism), ())) for accession, organism in jdp_organisms.items()
    }


def hsp70_index_by_genus(
    hsp70_names: Mapping[str, str],
    hsp70_organisms: Mapping[str, str],
) -> dict[str, list[str]]:
    """Which Hsp70 gene names occur in each genus."""
    index: dict[str, set[str]] = {}
    for accession, organism in hsp70_organisms.items():
        name = hsp70_names.get(accession)
        if not name:
            continue
        index.setdefault(genus_of(organism), set()).add(name)
    return {genus: sorted(names) for genus, names in index.items()}


def pairwise_class_contrast(
    classes: Mapping[str, str],
    partner_sets: Mapping[str, Sequence[str]],
    label: str,
    granularities: Sequence[str],
) -> dict[str, object]:
    """Which classes carry one partner label more than the others.

    The sweep says whether class and partner are associated; this says *how*, which is what
    a reader needs before believing it.
    """
    counts: dict[str, tuple[int, int]] = {}
    for accession, jdp_class in classes.items():
        genes = partner_sets.get(accession)
        if not genes:
            continue
        has = label in partner_labels(genes, granularities)
        carried, total = counts.get(jdp_class, (0, 0))
        counts[jdp_class] = (carried + int(has), total + 1)

    rates = {
        jdp_class: {"n": total, "carrying": carried, "rate": round(carried / total, 4) if total else 0.0}
        for jdp_class, (carried, total) in sorted(counts.items())
    }

    contrasts: dict[str, object] = {}
    for first, second in combinations(sorted(counts), 2):
        a_carried, a_total = counts[first]
        b_carried, b_total = counts[second]
        odds, p_value = fisher_exact([[a_carried, a_total - a_carried], [b_carried, b_total - b_carried]])
        contrasts[f"{first}_vs_{second}"] = {"odds_ratio": round(float(odds), 3), "p_value": float(p_value)}

    return {"label": label, "rates_by_class": rates, "contrasts": contrasts}
