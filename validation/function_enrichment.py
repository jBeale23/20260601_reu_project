"""Does the structural split predict function?

This is the experiment that decides whether the populations found inside classes B and C
are new *classes* or merely new *shapes*. Structural difference on its own reorganises
annotation; a difference that predicts what a protein does is a claim about biology.

The test
--------
For each functional term - a Gene Ontology term, a UniProt keyword, or a subcellular
location - ask whether it occurs at different rates in the two subpopulations, using
Fisher's exact test on the two-by-two table. Terms are ranked by adjusted p-value and by
odds ratio, so a term can be both reliable and large.

Three things keep this from being a well-studied-protein detector
----------------------------------------------------------------
**Annotated subset only.** Functional annotation concentrates in model organisms, exactly
as the class labels do. Comparing annotated against unannotated proteins would mostly
recover which ones somebody has studied. Every count here is taken within the subset that
carries annotation at all, and the coverage of each group is reported beside the result.

**Multiple testing.** Thousands of terms are tested at once, so raw p-values would produce
false positives in proportion to the vocabulary size. Benjamini-Hochberg adjustment is
applied across all terms tested in one comparison.

**A frequency floor.** A term appearing in three proteins can reach significance on a
two-by-two table while telling us nothing generalisable. Terms below a minimum count in
the combined set are not tested, which also keeps the multiple-testing correction from
being diluted by thousands of singletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.stats import fisher_exact

from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from data_fetching.fetch_function import FunctionalRecord

# Occurrences required in the combined annotated set before a term is tested. Below this a
# significant result is not generalisable and only dilutes the correction.
MIN_TERM_COUNT = 10

# Adjusted p-value below which a term is reported as differentially associated.
SIGNIFICANCE_FDR = 0.05

# Which annotation fields are tested, and what each is evidence of.
TERM_SOURCES = ("go_function", "go_process", "go_component", "keywords", "subcellular_locations")


@dataclass(frozen=True, slots=True)
class TermAssociation:
    """One functional term's association with a subpopulation."""

    term: str
    source: str
    count_in_group: int
    count_in_other: int
    n_group: int
    n_other: int
    odds_ratio: float
    p_value: float
    p_adjusted: float = 1.0

    @property
    def rate_in_group(self) -> float:
        """Share of the first group carrying this term."""
        return self.count_in_group / self.n_group if self.n_group else 0.0

    @property
    def rate_in_other(self) -> float:
        """Share of the second group carrying this term."""
        return self.count_in_other / self.n_other if self.n_other else 0.0

    @property
    def is_significant(self) -> bool:
        """Whether the association survives multiple-testing correction."""
        return self.p_adjusted < SIGNIFICANCE_FDR

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "term": self.term,
            "source": self.source,
            "count_in_group": self.count_in_group,
            "count_in_other": self.count_in_other,
            "rate_in_group": round(self.rate_in_group, 4),
            "rate_in_other": round(self.rate_in_other, 4),
            "odds_ratio": round(self.odds_ratio, 3),
            "p_value": self.p_value,
            "p_adjusted": round(self.p_adjusted, 6),
            "significant": self.is_significant,
        }


def _terms_for(record: FunctionalRecord, source: str) -> tuple[str, ...]:
    return tuple(getattr(record, source, ()) or ())


def compare_groups(
    group: Sequence[FunctionalRecord],
    other: Sequence[FunctionalRecord],
    *,
    sources: Sequence[str] = TERM_SOURCES,
    min_count: int = MIN_TERM_COUNT,
) -> list[TermAssociation]:
    """Test every functional term for differential association between two groups.

    Both groups must already be restricted to proteins carrying annotation; passing
    unannotated proteins in would make the test measure study effort rather than biology.
    """
    n_group, n_other = len(group), len(other)
    if n_group == 0 or n_other == 0:
        return []

    counts: dict[tuple[str, str], list[int]] = {}
    for records, slot in ((group, 0), (other, 1)):
        for record in records:
            for source in sources:
                for term in set(_terms_for(record, source)):
                    counts.setdefault((source, term), [0, 0])[slot] += 1

    associations: list[TermAssociation] = []
    for (source, term), (in_group, in_other) in counts.items():
        if in_group + in_other < min_count:
            continue
        table = [[in_group, n_group - in_group], [in_other, n_other - in_other]]
        odds, p_value = fisher_exact(table)
        associations.append(
            TermAssociation(
                term=term,
                source=source,
                count_in_group=in_group,
                count_in_other=in_other,
                n_group=n_group,
                n_other=n_other,
                odds_ratio=float(odds),
                p_value=float(p_value),
            ),
        )

    adjusted = benjamini_hochberg({f"{item.source}|{item.term}": item.p_value for item in associations})
    corrected = [
        TermAssociation(
            term=item.term,
            source=item.source,
            count_in_group=item.count_in_group,
            count_in_other=item.count_in_other,
            n_group=item.n_group,
            n_other=item.n_other,
            odds_ratio=item.odds_ratio,
            p_value=item.p_value,
            p_adjusted=adjusted[f"{item.source}|{item.term}"],
        )
        for item in associations
    ]
    corrected.sort(key=lambda item: (item.p_adjusted, -abs(item.rate_in_group - item.rate_in_other)))
    return corrected


def annotation_coverage(
    accessions: Sequence[str],
    functions: Mapping[str, FunctionalRecord],
) -> dict[str, object]:
    """How much of a group carries functional annotation at all.

    Reported beside every enrichment result, because a comparison between a well-annotated
    group and a sparsely annotated one measures study effort before it measures biology.
    """
    present = [functions[accession] for accession in accessions if accession in functions]
    annotated = [record for record in present if record.has_any_annotation]
    return {
        "n_proteins": len(accessions),
        "n_fetched": len(present),
        "n_annotated": len(annotated),
        "annotation_coverage": round(len(annotated) / len(accessions), 4) if accessions else 0.0,
        "median_go_terms": (sorted(record.n_go_terms for record in annotated)[len(annotated) // 2] if annotated else 0),
    }


@dataclass(frozen=True, slots=True)
class SubpopulationPair:
    """Two named groups of accessions to compare."""

    group_name: str
    other_name: str
    group_accessions: Sequence[str]
    other_accessions: Sequence[str]


def functional_split_report(
    pair: SubpopulationPair,
    functions: Mapping[str, FunctionalRecord],
    *,
    max_terms: int = 30,
) -> dict[str, object]:
    """Full comparison of two subpopulations, with coverage stated first."""
    group_name, other_name = pair.group_name, pair.other_name
    group_accessions, other_accessions = pair.group_accessions, pair.other_accessions
    group = [functions[a] for a in group_accessions if a in functions and functions[a].has_any_annotation]
    other = [functions[a] for a in other_accessions if a in functions and functions[a].has_any_annotation]

    associations = compare_groups(group, other)
    significant = [item for item in associations if item.is_significant]

    return {
        "comparison": f"{group_name} vs {other_name}",
        "coverage": {
            group_name: annotation_coverage(group_accessions, functions),
            other_name: annotation_coverage(other_accessions, functions),
        },
        "n_terms_tested": len(associations),
        "n_significant": len(significant),
        "significance_fdr": SIGNIFICANCE_FDR,
        "min_term_count": MIN_TERM_COUNT,
        "top_terms": [item.to_json_dict() for item in associations[:max_terms]],
        "interpretation": (
            "Terms are tested only within the annotated subset of each group, so this "
            "measures functional difference rather than difference in how well studied the "
            "two groups are. Compare the coverage figures before reading the terms: if they "
            "differ greatly, the annotated subsets are not comparable populations."
        ),
    }
