"""Choosing which proteins enter an all-versus-all comparison.

An all-versus-all over every AlphaFold model is 142,948 structures and roughly 10^10 pairs.
That is not merely expensive - staging the decompressed models for it exhausted a shared
group quota and killed two unrelated jobs - and it is not what the question needs. The
coupling analysis asks how sequence, structure and function move relative to one another,
and a set dominated by whichever families happen to be over-sequenced answers that question
about sequencing effort rather than about biology.

So a subset is chosen deliberately, and *how* it is chosen is itself a claim. Four
strategies are offered rather than one, because they answer subtly different questions and
the honest way to find out whether a result depends on the choice is to run more than one.

``architecture``
    One protein per InterPro domain architecture. The default, and the unit this project is
    built on: if the claim is that architecture predicts function, the natural denominator
    is one example per architecture, not one per sequenced genome. Roughly 4,155 of them.
``species`` / ``genus``
    One per taxon. Removes phylogenetic over-representation directly, which is the bias the
    architecture strategy only removes incidentally.
``random``
    A seeded sample. Unbiased in expectation and useful precisely because it ignores every
    structure in the data - if a result holds here and under ``architecture``, it is not an
    artefact of how representatives were picked.

Within a group the best-resolved member wins, not an arbitrary one: mean pLDDT where
structural features are available, falling back to the longest sequence. A representative
chosen at random would mean a low-confidence model could speak for its whole architecture.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.recurrence import genus_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.records import DomainStore

logger = logging.getLogger(__name__)

ARCHITECTURE = "architecture"
SPECIES = "species"
GENUS = "genus"
RANDOM = "random"
STRATEGIES = (ARCHITECTURE, SPECIES, GENUS, RANDOM)

DEFAULT_STRATEGY = ARCHITECTURE
DEFAULT_SEED = 0

# A scientific name is genus plus specific epithet.
_SPECIES_WORDS = 2


def species_of(organism_name: str) -> str:
    """First two words of a scientific name, which is the species."""
    parts = (organism_name or "").split()
    return " ".join(parts[:_SPECIES_WORDS]) if len(parts) >= _SPECIES_WORDS else (organism_name or "")


def _group_key(
    accession: str,
    store: DomainStore,
    strategy: str,
    rng: random.Random,
) -> str:
    """The group an accession belongs to under one strategy."""
    record = store.proteins[accession]
    if strategy == ARCHITECTURE:
        # An accession with no recorded architecture is its own group rather than joining a
        # single enormous "unknown" bucket that one protein would then represent.
        return record.ida_accession or f"__no_ida__{accession}"
    if strategy == SPECIES:
        return species_of(record.organism_name) or f"__no_organism__{accession}"
    if strategy == GENUS:
        return genus_of(record.organism_name) or f"__no_organism__{accession}"
    if strategy == RANDOM:
        return f"__random__{rng.random()}"
    message = f"unknown strategy: {strategy}"
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """How representatives are chosen, bundled so the strategy travels with its knobs."""

    strategy: str = DEFAULT_STRATEGY
    limit: int | None = None
    seed: int = DEFAULT_SEED


DEFAULT_SELECTION = SelectionOptions()


def choose_representatives(
    store: DomainStore,
    *,
    options: SelectionOptions = DEFAULT_SELECTION,
    quality: Mapping[str, float] | None = None,
    restrict_to: Sequence[str] | None = None,
) -> list[str]:
    """One accession per group, best-resolved member first.

    Args:
        store: The domain store to draw from.
        options: Strategy, optional cap, and seed.
        quality: Optional per-accession score - mean pLDDT - deciding which member of a
            group represents it. Absent, the longest sequence wins.
        restrict_to: Optional accessions to consider at all, e.g. those with a model.

    Returns:
        Accessions, sorted, one per group.
    """
    strategy, limit = options.strategy, options.limit
    if strategy not in STRATEGIES:
        message = f"unknown strategy {strategy!r}; expected one of {STRATEGIES}"
        raise ValueError(message)

    rng = random.Random(options.seed)  # noqa: S311 - sampling a dataset, not generating a secret
    allowed = set(restrict_to) if restrict_to is not None else None
    scores = quality or {}

    # Two scores per candidate, kept apart on purpose. Mean pLDDT and sequence length are
    # on incomparable scales, so mixing them into one number is only safe *within* a group -
    # where every member either has a structural score or none do. Ranking across groups on
    # a mixed number silently let a long unscored protein outrank a well-resolved one.
    best: dict[str, tuple[float, float, str]] = {}
    for accession, record in store.proteins.items():
        if allowed is not None and accession not in allowed:
            continue
        key = _group_key(accession, store, strategy, rng)
        structural = float(scores[accession]) if accession in scores else float("-inf")
        fallback = float(len(record.sequence))
        within = structural if structural != float("-inf") else fallback
        current = best.get(key)
        current_within = current[0] if current else float("-inf")
        if current is None or within > current_within or (within == current_within and accession < current[2]):
            best[key] = (within, structural, accession)

    chosen = sorted(item[2] for item in best.values())
    if limit is not None and len(chosen) > limit:
        # The cap ranks on the structural score alone, so it compares like with like. A
        # representative with no structural score sorts last, which is the right direction
        # for a structural comparison: it has no model to contribute.
        ranked = sorted(best.values(), key=lambda item: (-item[1], item[2]))[:limit]
        chosen = sorted(accession for _within, _structural, accession in ranked)
    logger.info(
        "%s strategy: %s representative(s) from %s protein(s)",
        strategy,
        len(chosen),
        len(allowed) if allowed is not None else len(store.proteins),
    )
    return chosen


def representative_summary(
    store: DomainStore,
    chosen: Sequence[str],
    *,
    strategy: str,
) -> dict[str, object]:
    """What a representative set covers, for the report.

    Coverage is part of the result. A set of 4,155 representatives standing for 181,526
    proteins is a different claim from one standing for 5,000, and the ratio belongs beside
    any conclusion drawn from it.
    """
    architectures = {store.proteins[a].ida_accession for a in chosen if a in store.proteins}
    organisms = {store.proteins[a].organism_name for a in chosen if a in store.proteins}
    return {
        "strategy": strategy,
        "n_representatives": len(chosen),
        "n_proteins_in_store": len(store.proteins),
        "n_architectures_covered": len(architectures - {""}),
        "n_species_covered": len({species_of(name) for name in organisms} - {""}),
        "n_genera_covered": len({genus_of(name) for name in organisms} - {""}),
        "note": (
            "An all-versus-all over every model is roughly 10^10 pairs and, staged, exceeds "
            "the shared quota. A representative set is chosen instead, and the strategy is "
            "reported because it is a claim: 'architecture' asks the question per domain "
            "architecture, 'species'/'genus' per taxon, 'random' ignores both. A result "
            "holding under more than one is not an artefact of the choice."
        ),
    }
