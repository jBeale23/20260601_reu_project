"""Do J-domain protein classes fail in characteristic ways?

A chaperone class is defined here by what its members are built like. The question this
module asks is whether that structural definition predicts anything about *pathology* -
whether the diseases caused by losing a class's members cluster, and cluster differently
from those caused by losing another class's.

If they do, the classification is not merely a tidy way to sort sequences. It would mean
architecture constrains which cellular process a co-chaperone is load-bearing for, and that
losing it fails that process specifically. J-domain proteins already hint at this: DNAJB6
causes a limb-girdle muscular dystrophy, DNAJC5 an adult neuronal ceroid lipofuscinosis,
DNAJC19 a dilated cardiomyopathy, SACS an early-onset ataxia. Those are different organs
failing, from proteins in the same superfamily.

What this can and cannot show
-----------------------------
Curated disease associations exist almost exclusively on reviewed entries, and reviewed
entries are heavily human. So this measures which *human* J-domain proteins have been
linked to disease, which is a statement about clinical genetics and study effort as much as
about biology. Two guards follow from that: the number of proteins carrying any disease
association is reported beside every result, and nothing is concluded from a class with
fewer than a handful of them.

Disease grouping is by organ system and mechanism rather than by individual disease,
because individual diseases are nearly all singletons here - one protein, one syndrome -
and a test over singletons has no power. The groups below are broad on purpose.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping

    from data_fetching.fetch_function import FunctionalRecord

logger = logging.getLogger(__name__)

# Organ system and mechanism groups, matched on lowercased disease names.
#
# A disease belongs to *every* group it matches, not the first. Many of these syndromes are
# genuinely multi-organ - dilated cardiomyopathy with ataxia (DNAJC19) fails a heart and a
# cerebellum - and forcing such a disease into one bucket discards the fact that makes it
# interesting. Multi-label also removes the ordering sensitivity that first-match-wins has.
DISEASE_GROUPS: dict[str, tuple[str, ...]] = {
    "neuromuscular": (
        "muscular dystrophy",
        "myasthen",
        "motor neuropathy",
        "charcot-marie-tooth",
        "spastic paraplegia",
        "amyotrophic",
    ),
    "neurodegenerative": (
        "ceroid lipofuscinosis",
        "parkinson",
        "alzheimer",
        "ataxia",
        "neurodegener",
        "dementia",
        "huntington",
        "spinocerebellar",
    ),
    "neurodevelopmental": (
        "intellectual disability",
        "mental retardation",
        "microcephal",
        "developmental and epileptic",
        "autism",
        "epileptic encephalopathy",
    ),
    "cardiac": ("cardiomyopathy", "cardiac", "long qt", "heart"),
    "renal": ("kidney", "renal", "nephro", "polycystic"),
    "ciliopathy": ("ciliary dyskinesia", "ciliopath", "bardet-biedl", "situs inversus"),
    "ophthalmic": ("retinitis", "retinal", "cataract", "macular", "blindness", "optic atrophy"),
    "metabolic": ("diabetes", "hyperphenylalaninemia", "metabolic", "obesity", "lipodystrophy"),
    "immune_haematologic": ("immunodeficiency", "anemia", "anaemia", "neutropenia", "thrombocytopenia"),
    "cancer": ("cancer", "carcinoma", "tumor", "tumour", "leukemia", "leukaemia", "melanoma", "sarcoma"),
    "skeletal_connective": ("skeletal dysplasia", "osteogenesis", "chondrodysplasia", "arthrogryposis"),
    "dermatologic": ("ichthyosis", "epidermolysis", "keratoderma", "alopecia"),
}

UNGROUPED = "other"

# A class needs at least this many disease-associated members before a rate is quoted.
# Below it, one protein moves the rate by tens of percentage points.
MIN_DISEASE_PROTEINS_PER_CLASS = 3

SIGNIFICANCE_LEVEL = 0.05


# "myopathy" is a substring of "cardiomyopathy", so a plain containment test filed every
# dilated cardiomyopathy under neuromuscular. Skeletal-muscle myopathy is matched only when
# the word is not part of a cardiac compound.
_MYOPATHY_EXCLUSIONS = ("cardiomyopathy", "cardio-myopathy")


def disease_groups(name: str) -> set[str]:
    """Every organ-system or mechanism group a disease name belongs to.

    A set, not a single label: these syndromes are often multi-organ, and the ones that are
    - a cardiomyopathy presenting with ataxia, say - are exactly the informative cases.
    """
    lowered = (name or "").lower()
    found = {group for group, patterns in DISEASE_GROUPS.items() if any(p in lowered for p in patterns)}
    if "myopathy" in lowered and not any(token in lowered for token in _MYOPATHY_EXCLUSIONS):
        found.add("neuromuscular")
    return found or {UNGROUPED}


def groups_for(record: FunctionalRecord) -> set[str]:
    """Every disease group a protein is linked to, across all of its diseases."""
    return {group for name in record.diseases for group in disease_groups(name)}


@dataclass(frozen=True, slots=True)
class DiseaseAssociation:
    """Whether one class is linked to one disease group more than the others are."""

    disease_group: str
    protein_class: str
    n_in_class: int
    n_class_total: int
    n_in_others: int
    n_others_total: int
    p_value: float
    p_adjusted: float = 1.0
    examples: tuple[str, ...] = ()

    @property
    def rate_in_class(self) -> float:
        """Share of the class's disease-linked proteins tied to this group."""
        return self.n_in_class / self.n_class_total if self.n_class_total else 0.0

    @property
    def rate_in_others(self) -> float:
        """The same share across every other class."""
        return self.n_in_others / self.n_others_total if self.n_others_total else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "disease_group": self.disease_group,
            "class": self.protein_class,
            "n_in_class": self.n_in_class,
            "n_class_total": self.n_class_total,
            "rate_in_class": round(self.rate_in_class, 4),
            "rate_in_others": round(self.rate_in_others, 4),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "examples": list(self.examples),
        }


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for a 2x2 table.

    Exact rather than chi-square because the counts here are small - a class may have five
    disease-linked members - and chi-square is unreliable in exactly that regime.
    """
    total = a + b + c + d
    if total == 0:
        return 1.0
    row1, col1 = a + b, a + c
    low = max(0, col1 - (total - row1))
    high = min(row1, col1)
    denominator = comb(total, col1)
    observed = comb(row1, a) * comb(total - row1, col1 - a)
    tail = sum(
        comb(row1, value) * comb(total - row1, col1 - value)
        for value in range(low, high + 1)
        if comb(row1, value) * comb(total - row1, col1 - value) <= observed
    )
    return min(1.0, tail / denominator) if denominator else 1.0


def disease_associations(
    classes: Mapping[str, str],
    records: Mapping[str, FunctionalRecord],
) -> list[DiseaseAssociation]:
    """Test every class against every disease group, corrected together."""
    linked = {
        accession: groups_for(record)
        for accession, record in records.items()
        if accession in classes and record.has_disease_association
    }
    if not linked:
        return []

    by_class: dict[str, list[str]] = {}
    for accession in linked:
        by_class.setdefault(classes[accession], []).append(accession)

    groups = sorted({group for found in linked.values() for group in found})
    results: list[DiseaseAssociation] = []
    for group in groups:
        carriers = {accession for accession, found in linked.items() if group in found}
        for protein_class, members in by_class.items():
            if len(members) < MIN_DISEASE_PROTEINS_PER_CLASS:
                continue
            in_class = len(carriers & set(members))
            others = [a for a in linked if classes[a] != protein_class]
            in_others = len(carriers & set(others))
            results.append(
                DiseaseAssociation(
                    disease_group=group,
                    protein_class=protein_class,
                    n_in_class=in_class,
                    n_class_total=len(members),
                    n_in_others=in_others,
                    n_others_total=len(others),
                    p_value=_fisher_two_sided(
                        in_class,
                        len(members) - in_class,
                        in_others,
                        len(others) - in_others,
                    ),
                    examples=tuple(sorted(carriers & set(members))[:5]),
                ),
            )

    adjusted = benjamini_hochberg({f"{r.disease_group}|{r.protein_class}": r.p_value for r in results})
    return [
        DiseaseAssociation(
            disease_group=r.disease_group,
            protein_class=r.protein_class,
            n_in_class=r.n_in_class,
            n_class_total=r.n_class_total,
            n_in_others=r.n_in_others,
            n_others_total=r.n_others_total,
            p_value=r.p_value,
            p_adjusted=adjusted[f"{r.disease_group}|{r.protein_class}"],
            examples=r.examples,
        )
        for r in results
    ]


def disease_report(
    classes: Mapping[str, str],
    records: Mapping[str, FunctionalRecord],
    *,
    max_reported: int = 40,
) -> dict[str, object]:
    """Whether class predicts which organ system fails.

    Coverage leads the report. Disease annotation exists almost only on reviewed entries,
    and a result drawn from a handful of human proteins is a different claim from one drawn
    from hundreds - so the counts come before the p-values, not after them.
    """
    shared = {a: r for a, r in records.items() if a in classes}
    linked = {a: r for a, r in shared.items() if r.has_disease_association}
    reviewed = sum(1 for r in shared.values() if r.is_reviewed)

    group_counts: Counter[str] = Counter()
    disease_names: Counter[str] = Counter()
    for record in linked.values():
        for name in record.diseases:
            for group in disease_groups(name):
                group_counts[group] += 1
            disease_names[name] += 1

    by_class: Counter[str] = Counter(classes[a] for a in linked)
    associations = disease_associations(classes, records)
    associations.sort(key=lambda item: (item.p_adjusted, -item.rate_in_class))

    return {
        "n_classified_with_annotation": len(shared),
        "n_reviewed": reviewed,
        "n_with_disease_association": len(linked),
        "disease_linked_by_class": dict(by_class.most_common()),
        "diseases_by_group": dict(group_counts.most_common()),
        "n_distinct_diseases": len(disease_names),
        "n_associations_tested": len(associations),
        "n_significant": sum(1 for item in associations if item.p_adjusted < SIGNIFICANCE_LEVEL),
        "associations": [item.to_json_dict() for item in associations[:max_reported]],
        "caveat": (
            "Curated disease associations sit almost exclusively on reviewed entries, which "
            "are heavily human, so this measures which human J-domain proteins have been "
            "linked to disease - part biology, part clinical study effort. Diseases are "
            "grouped by organ system because individual diseases here are nearly all "
            "singletons, and a test over singletons has no power. A class with fewer than "
            f"{MIN_DISEASE_PROTEINS_PER_CLASS} disease-linked members is not tested."
        ),
    }
