"""Permutation null for the novel-category call.

The novelty score is a weighted sum of hand-chosen terms, so "1,613 candidates" means
nothing until we know how many the same rules would flag on data with the biological
signal removed. This module answers that by rebuilding the score on permuted evidence.

Choosing a null that can actually fail
--------------------------------------
The obvious null — shuffle whole architectures between proteins — is worthless here, and
it is worth being explicit about why. The candidate count is a function of the *multiset*
of architectures present, and shuffling whole architectures preserves that multiset
exactly, so the null reproduces the observed count no matter how much or how little real
signal exists. A test that cannot fail is not evidence.

The default null is therefore a configuration model over the protein-to-partner-domain
bipartite graph (:func:`permute_partner_domains`): each protein keeps the number of
partner domains it has, each domain family keeps its total frequency, and only the
pairing is randomised. That can fail, because a protein with several partners is more
likely to receive a characterised one by chance — so if uncharacterised partners really
do concentrate on particular proteins, the real data will exceed the null.

Sequence-derived evidence (HPD presence, G/F-rich composition, J-domain placement) is
never shuffled: it belongs to the protein, not to its domain arrangement.

A caveat the report states plainly: when every protein carries exactly one partner
domain, the candidate count is fixed by the marginal frequencies alone and *no*
permutation scheme can show enrichment. In that regime the novelty flag is a descriptive
filter over architecture, and the cross-species recurrence test carries the statistical
weight instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from domain_layout.constants import (
    FAMILY_DNAJ_C,
    FAMILY_J_DOMAIN,
    FAMILY_OTHER,
    FAMILY_ZINC_FINGER,
    NOVELTY_THRESHOLD,
)
from domain_layout.profiles import classify_layout

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    PermutationStrategy = Callable[[Sequence["LayoutEvidence"], random.Random], list["LayoutEvidence"]]

    from domain_layout.profiles import LayoutEvidence
    from domain_layout.shark import SharkMatch

# Fields describing the domain arrangement. These move together during a permutation:
# splitting them would create architectures that never occur in nature and make the null
# easier to beat than it should be.
ARCHITECTURE_FIELDS = (
    "has_dnaj_c",
    "has_zinc_finger_like",
    "has_gf_rich_region",
    "j_domain_position",
    "n_structured_domains",
    "unknown_partner_families",
    "domain_family_layout",
    "structured_families",
)


@dataclass(frozen=True, slots=True)
class NullResult:
    """Observed candidate rate compared with its permutation null."""

    n_proteins: int
    observed_candidates: int
    null_mean: float
    null_std: float
    null_min: int
    null_max: int
    p_value: float
    n_permutations: int

    @property
    def observed_rate(self) -> float:
        """Fraction of proteins flagged in the real data."""
        return self.observed_candidates / self.n_proteins if self.n_proteins else 0.0

    @property
    def enrichment(self) -> float:
        """Observed candidates divided by the null expectation."""
        return self.observed_candidates / self.null_mean if self.null_mean > 0 else 0.0

    @property
    def empirical_fdr(self) -> float:
        """Share of the observed calls the null alone would account for.

        This is a false-discovery *rate* in the practical sense: if the null produces
        800 calls where the data produces 1,000, then at least 80% of the calls carry no
        more information than the shuffled data does.
        """
        if self.observed_candidates <= 0:
            return 1.0
        return min(1.0, self.null_mean / self.observed_candidates)

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "n_proteins": self.n_proteins,
            "n_permutations": self.n_permutations,
            "observed_candidates": self.observed_candidates,
            "observed_rate": round(self.observed_rate, 5),
            "null_mean_candidates": round(self.null_mean, 2),
            "null_std": round(self.null_std, 2),
            "null_range": [self.null_min, self.null_max],
            "enrichment_over_null": round(self.enrichment, 3),
            "empirical_fdr": round(self.empirical_fdr, 4),
            "p_value": round(self.p_value, 5),
        }


def permute_architecture_evidence(
    evidence: Sequence[LayoutEvidence],
    rng: random.Random,
) -> list[LayoutEvidence]:
    """Reassign whole architectures between proteins, keeping sequence evidence.

    Retained for comparison, but note what it cannot show: the candidate count is a
    function of the multiset of architectures, which this permutation preserves exactly.
    It therefore reproduces the observed count almost identically no matter how much real
    signal exists. :func:`permute_partner_domains` is the null with power.
    """
    order = list(range(len(evidence)))
    rng.shuffle(order)
    return [
        replace(
            evidence[index],
            **{field: getattr(evidence[donor], field) for field in ARCHITECTURE_FIELDS},
        )
        for index, donor in enumerate(order)
    ]


def partner_families(item: LayoutEvidence) -> tuple[str, ...]:
    """Structured families accompanying the J-domain."""
    return tuple(family for family in item.structured_families if family != FAMILY_J_DOMAIN)


def _rebuild_from_partners(item: LayoutEvidence, partners: Sequence[str]) -> LayoutEvidence:
    """Recompute architecture-derived evidence from a new partner-domain multiset."""
    unique = tuple(dict.fromkeys(partners))
    families = (FAMILY_J_DOMAIN, *unique) if item.has_j_domain else unique
    return replace(
        item,
        has_dnaj_c=FAMILY_DNAJ_C in partners,
        has_zinc_finger_like=FAMILY_ZINC_FINGER in partners,
        unknown_partner_families=sum(1 for family in partners if family == FAMILY_OTHER),
        n_structured_domains=(1 if item.has_j_domain else 0) + len(partners),
        structured_families=families,
        domain_family_layout=">".join(families),
    )


def permute_partner_domains(
    evidence: Sequence[LayoutEvidence],
    rng: random.Random,
) -> list[LayoutEvidence]:
    """Redistribute partner domains between proteins, preserving both degree sequences.

    This is a configuration model over the protein-to-domain-family bipartite graph:
    every protein keeps the *number* of partner domains it really has, and every domain
    family keeps its total frequency across the dataset, but which protein carries which
    partner is randomised.

    That is the null the novelty claim needs. It asks whether proteins whose only
    partners are uncharacterised occur more often than the marginal frequencies of those
    domain families would predict — i.e. whether unusual partners genuinely cluster on
    the same proteins, rather than being spread as chance would spread them.

    Sequence-derived evidence (HPD, G/F-rich composition, J-domain placement) stays with
    its own protein, since nothing about it depends on the partner shuffle.
    """
    pool: list[str] = []
    counts: list[int] = []
    for item in evidence:
        families = partner_families(item)
        counts.append(len(families))
        pool.extend(families)

    rng.shuffle(pool)

    permuted: list[LayoutEvidence] = []
    cursor = 0
    for item, count in zip(evidence, counts, strict=True):
        assigned = pool[cursor : cursor + count]
        cursor += count
        permuted.append(_rebuild_from_partners(item, assigned))
    return permuted


def count_candidates(
    evidence: Sequence[LayoutEvidence],
    matches: Sequence[SharkMatch | None],
) -> int:
    """How many proteins the current rules would flag as novel-category candidates."""
    return sum(
        1
        for item, match in zip(evidence, matches, strict=True)
        if classify_layout(item, shark_match=match).novel_class_candidate
    )


def novelty_null(
    evidence: Sequence[LayoutEvidence],
    matches: Sequence[SharkMatch | None],
    *,
    n_permutations: int = 200,
    seed: int = 0,
    permute: PermutationStrategy = permute_partner_domains,
) -> NullResult:
    """Compare the observed candidate count with its permutation null.

    Defaults to the configuration-model null over partner domains, which is the variant
    with power; pass ``permute=permute_architecture_evidence`` to see the degenerate
    whole-architecture shuffle for comparison.

    The p-value is the one-sided probability of the null producing at least as many
    candidates as the real data, with the usual +1 correction so a finite number of
    permutations can never yield exactly zero.
    """
    observed = count_candidates(evidence, matches)
    n_proteins = len(evidence)
    if n_proteins == 0 or n_permutations <= 0:
        return NullResult(
            n_proteins=n_proteins,
            observed_candidates=observed,
            null_mean=0.0,
            null_std=0.0,
            null_min=0,
            null_max=0,
            p_value=1.0,
            n_permutations=max(0, n_permutations),
        )

    # Deterministic, seeded permutations for reproducibility; not a security context.
    rng = random.Random(seed)  # noqa: S311
    counts: list[int] = []
    for _ in range(n_permutations):
        permuted = permute(evidence, rng)
        counts.append(count_candidates(permuted, matches))

    mean = sum(counts) / len(counts)
    variance = sum((value - mean) ** 2 for value in counts) / len(counts)
    at_least = sum(1 for value in counts if value >= observed)

    return NullResult(
        n_proteins=n_proteins,
        observed_candidates=observed,
        null_mean=mean,
        null_std=variance**0.5,
        null_min=min(counts),
        null_max=max(counts),
        p_value=(at_least + 1) / (n_permutations + 1),
        n_permutations=n_permutations,
    )


def score_threshold_sweep(
    evidence: Sequence[LayoutEvidence],
    matches: Sequence[SharkMatch | None],
    *,
    thresholds: Sequence[float] = (0.35, 0.5, 0.65, 0.8),
) -> list[dict[str, object]]:
    """Candidate counts across novelty thresholds.

    The default threshold is a judgement call, so the report shows how sensitive the
    candidate set is to it rather than presenting one cut as though it were derived.
    """
    scores = [
        classify_layout(item, shark_match=match).novelty_score for item, match in zip(evidence, matches, strict=True)
    ]
    has_j_domain = [item.has_j_domain for item in evidence]
    return [
        {
            "threshold": threshold,
            "n_candidates": sum(
                1 for score, has_j in zip(scores, has_j_domain, strict=True) if has_j and score >= threshold
            ),
            "is_default": threshold == NOVELTY_THRESHOLD,
        }
        for threshold in thresholds
    ]
