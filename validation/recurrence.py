"""Cross-species recurrence of domain architectures, with a permutation null.

A single protein with an unusual architecture is unremarkable: annotation error,
incomplete gene model, or a genuine one-off. The same architecture appearing
independently in distant lineages is much harder to explain that way, so taxonomic
spread is the strongest evidence available here that a layout reflects a real, conserved
arrangement rather than noise.

Spread on its own is still not enough, because common architectures appear in many taxa
simply by being common. The permutation null below breaks the link between architecture
and organism while holding both marginal distributions fixed, which answers the sharper
question: *given how often this architecture occurs at all, is it spread across more
genera than chance would produce?*
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Genus is the taxonomic unit: it is derivable from the organism name already stored with
# every record, needs no extra taxonomy service, and is coarse enough that repetition
# across genera implies real evolutionary distance rather than strain-level duplicates.
UNKNOWN_GENUS = "unknown"


def genus_of(organism_name: str) -> str:
    """First token of a binomial name, used as the taxonomic unit."""
    token = (organism_name or "").strip().split(" ")[0] if organism_name else ""
    return token or UNKNOWN_GENUS


@dataclass(frozen=True, slots=True)
class ArchitectureRecurrence:
    """How widely one architecture is distributed across taxa."""

    architecture: str
    n_proteins: int
    n_genera: int
    n_organisms: int
    example_accessions: tuple[str, ...]
    null_mean_genera: float = 0.0
    p_value: float = 1.0

    @property
    def genera_enrichment(self) -> float:
        """Observed genera divided by the null expectation (1.0 means as expected)."""
        return self.n_genera / self.null_mean_genera if self.null_mean_genera > 0 else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "architecture": self.architecture,
            "n_proteins": self.n_proteins,
            "n_genera": self.n_genera,
            "n_organisms": self.n_organisms,
            "null_mean_genera": round(self.null_mean_genera, 3),
            "genera_enrichment": round(self.genera_enrichment, 3),
            "p_value": round(self.p_value, 5),
            "example_accessions": list(self.example_accessions),
        }


def observed_recurrence(
    architectures: Mapping[str, str],
    organisms: Mapping[str, str],
    *,
    max_examples: int = 3,
) -> dict[str, ArchitectureRecurrence]:
    """Count proteins, distinct organisms, and distinct genera per architecture."""
    by_architecture: dict[str, list[str]] = defaultdict(list)
    for accession, architecture in architectures.items():
        by_architecture[architecture].append(accession)

    recurrence: dict[str, ArchitectureRecurrence] = {}
    for architecture, accessions in by_architecture.items():
        names = [organisms.get(accession, "") for accession in accessions]
        recurrence[architecture] = ArchitectureRecurrence(
            architecture=architecture,
            n_proteins=len(accessions),
            n_genera=len({genus_of(name) for name in names}),
            n_organisms=len({name for name in names if name}),
            example_accessions=tuple(sorted(accessions)[:max_examples]),
        )
    return recurrence


def _null_genera_counts(
    architecture_labels: Sequence[str],
    genera: Sequence[str],
    rng: random.Random,
) -> dict[str, int]:
    """One permutation: reassign architectures to proteins, keeping both marginals."""
    shuffled = list(architecture_labels)
    rng.shuffle(shuffled)
    seen: dict[str, set[str]] = defaultdict(set)
    for architecture, genus in zip(shuffled, genera, strict=True):
        seen[architecture].add(genus)
    return {architecture: len(values) for architecture, values in seen.items()}


def recurrence_with_null(
    architectures: Mapping[str, str],
    organisms: Mapping[str, str],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, ArchitectureRecurrence]:
    """Score each architecture's taxonomic spread against a permutation null.

    The null shuffles which protein carries which architecture while leaving each
    protein's organism untouched. Architecture frequencies and the taxonomic composition
    of the dataset are therefore both preserved exactly, and the only thing destroyed is
    the association between them.

    The reported p-value is the one-sided probability of seeing at least the observed
    number of distinct genera under that null, with the standard +1 correction so it can
    never be reported as exactly zero from a finite number of permutations.
    """
    observed = observed_recurrence(architectures, organisms)
    if not observed or n_permutations <= 0:
        return observed

    accessions = sorted(architectures)
    labels = [architectures[accession] for accession in accessions]
    genera = [genus_of(organisms.get(accession, "")) for accession in accessions]

    # Deterministic, seeded permutations for reproducibility; not a security context.
    rng = random.Random(seed)  # noqa: S311
    totals: Counter[str] = Counter()
    at_least_observed: Counter[str] = Counter()
    for _ in range(n_permutations):
        null_counts = _null_genera_counts(labels, genera, rng)
        for architecture, record in observed.items():
            null_value = null_counts.get(architecture, 0)
            totals[architecture] += null_value
            if null_value >= record.n_genera:
                at_least_observed[architecture] += 1

    return {
        architecture: ArchitectureRecurrence(
            architecture=record.architecture,
            n_proteins=record.n_proteins,
            n_genera=record.n_genera,
            n_organisms=record.n_organisms,
            example_accessions=record.example_accessions,
            null_mean_genera=totals[architecture] / n_permutations,
            p_value=(at_least_observed[architecture] + 1) / (n_permutations + 1),
        )
        for architecture, record in observed.items()
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate).

    One test is run per architecture, so raw p-values would produce false positives at a
    rate proportional to the number of architectures examined.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 1.0
    for rank, (key, value) in enumerate(reversed(ordered), start=1):
        index = total - rank + 1
        candidate = min(previous, value * total / index)
        adjusted[key] = min(1.0, candidate)
        previous = adjusted[key]
    return adjusted
