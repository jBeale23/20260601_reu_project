"""Interaction evidence from BioGRID.

Every other source in this project is homology or annotation. BioGRID is neither: it
records experiments in which two proteins were shown to interact, physically or
genetically. That makes it independent of sequence, of structure, and of the domain
signatures the class scheme is built on - so an interaction difference between two
subpopulations cannot be an artefact of how they were annotated.

Two kinds of evidence, and they answer different questions
----------------------------------------------------------
**Physical** interactions name the partner directly. For a J-domain protein the most
informative fact is usually which Hsp70 it works with, and whether it works with one at all.

**Genetic** interactions - synthetic lethality, suppression - say that two genes matter to
the same process without either protein touching the other. Yeast genetics contributes most
of these, and they reach functional relationships that no physical assay would show.

What this cannot fix
--------------------
BioGRID is as concentrated in model organisms as everything else: yeast, human, fly, worm.
It adds evidence of a different *kind*, not evidence about more organisms. Coverage is
therefore reported per group exactly as it is for Gene Ontology terms, and a comparison
between groups with very different coverage is not interpretable.

Access
------
BioGRID's REST interface requires a free access key, supplied through ``BIOGRID_ACCESS_KEY``.
Without one this module reports itself unavailable and the pipeline continues without
interaction evidence rather than failing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from data_fetching.utils import get_with_retry

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

BIOGRID_URL = "https://webservice.thebiogrid.org/interactions/"
ACCESS_KEY_VARIABLE = "BIOGRID_ACCESS_KEY"

# Proteins requested per call.
#
# One, not many. BioGRID accepts a gene list, but the response is a flat row set with a
# ceiling, and a single well-studied protein overruns it alone: YDJ1 returns 1,511 rows
# covering 1,011 distinct partners. Batching a hundred genes truncates the response and
# leaves almost every gene with a handful of rows - a first run gave five query proteins
# one or two partners each and missed every one of their Hsp70 partners, which for a
# co-chaperone is the interaction that matters most. Throughput comes from concurrency
# instead.
BATCH_SIZE = 1

DEFAULT_CONCURRENCY = 6

# Rows per request. BioGRID's documented hard cap: values above 10,000 are ignored, and
# retrieving more requires paging with `start`. A well-connected protein really can exceed
# it - YDJ1 alone returns 1,511 rows - so the pager below is not hypothetical.
MAX_ROWS_PER_BATCH = 10000

# Requests before paging is abandoned for one gene. At 10,000 rows each this allows a
# million interactions, far beyond any real protein, and stops a malformed response from
# looping forever.
MAX_PAGES = 100

PHYSICAL = "physical"
GENETIC = "genetic"


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """Interaction evidence for one protein."""

    identifier: str
    partners: tuple[str, ...] = ()
    physical_partners: tuple[str, ...] = ()
    genetic_partners: tuple[str, ...] = ()
    hsp70_partners: tuple[str, ...] = ()

    @property
    def n_partners(self) -> int:
        """Distinct interaction partners of any kind."""
        return len(self.partners)

    @property
    def has_hsp70_partner(self) -> bool:
        """Whether any partner is an Hsp70 family member.

        The single most informative fact about a J-domain protein: a co-chaperone that
        never partners an Hsp70 is doing something else.
        """
        return bool(self.hsp70_partners)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for the store."""
        return {
            "identifier": self.identifier,
            "partners": list(self.partners),
            "physical_partners": list(self.physical_partners),
            "genetic_partners": list(self.genetic_partners),
            "hsp70_partners": list(self.hsp70_partners),
        }


@dataclass(slots=True)
class InteractionStore:
    """Interaction records keyed by the identifier used to request them."""

    records: dict[str, InteractionRecord] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of records held."""
        return len(self.records)

    def __iter__(self) -> Iterator[InteractionRecord]:
        """Iterate records in identifier order."""
        return iter([self.records[key] for key in sorted(self.records)])

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the whole store."""
        return {
            "records": {key: value.to_json_dict() for key, value in self.records.items()},
            "failures": dict(self.failures),
        }


# Gene-name prefixes identifying Hsp70 family members across the organisms BioGRID covers.
_HSP70_PREFIXES = ("HSPA", "HSP70", "SSA", "SSB", "SSC", "SSE", "KAR2", "DNAK", "BIP", "HSPH")


def _is_hsp70(gene: str) -> bool:
    upper = gene.upper()
    return any(upper.startswith(prefix) for prefix in _HSP70_PREFIXES)


def biogrid_available() -> bool:
    """Whether an access key is configured."""
    return bool(os.environ.get(ACCESS_KEY_VARIABLE))


def record_from_interactions(identifier: str, payload: Mapping[str, Any] | list[Any]) -> InteractionRecord:
    """Collapse BioGRID's per-interaction rows into one record for a protein.

    BioGRID returns a mapping of row id to interaction when there are results and an empty
    list when there are none, so both shapes are accepted rather than assuming the first.
    """
    rows = payload.values() if isinstance(payload, dict) else (payload or [])
    # A batched request returns one payload covering every gene in the batch, so each row
    # must be matched back to the protein it belongs to. Without that check a row for one
    # gene contributes a partner to every gene in the batch - a first run gave all five
    # query proteins the same six partners and none of their real Hsp70 partners.
    physical: list[str] = []
    genetic: list[str] = []
    wanted = identifier.upper()
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        a = str(entry.get("OFFICIAL_SYMBOL_A", ""))
        b = str(entry.get("OFFICIAL_SYMBOL_B", ""))
        if wanted == a.upper():
            partner = b
        elif wanted == b.upper():
            partner = a
        else:
            continue
        if not partner or partner.upper() == wanted:
            # Self-interactions are real but say nothing about partner specificity.
            continue
        if str(entry.get("EXPERIMENTAL_SYSTEM_TYPE", "")).lower() == GENETIC:
            genetic.append(partner)
        else:
            physical.append(partner)

    everything = tuple(dict.fromkeys(physical + genetic))
    return InteractionRecord(
        identifier=identifier,
        partners=everything,
        physical_partners=tuple(dict.fromkeys(physical)),
        genetic_partners=tuple(dict.fromkeys(genetic)),
        hsp70_partners=tuple(gene for gene in everything if _is_hsp70(gene)),
    )


async def fetch_interactions(
    session: aiohttp.ClientSession,
    identifiers: Sequence[str],
    *,
    access_key: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> InteractionStore:
    """Fetch interaction evidence for a set of gene identifiers.

    Returns an empty store when no access key is configured, so a pipeline without one
    continues rather than failing.
    """
    key = access_key or os.environ.get(ACCESS_KEY_VARIABLE)
    store = InteractionStore()
    if not key:
        logger.warning("%s is not set; interaction evidence will not be fetched.", ACCESS_KEY_VARIABLE)
        return store

    semaphore = asyncio.Semaphore(max(1, concurrency))
    batches = [list(identifiers[start : start + BATCH_SIZE]) for start in range(0, len(identifiers), BATCH_SIZE)]

    async def worker(batch: Sequence[str]) -> None:
        query = "|".join(batch)
        # includeInteractors=true returns every interaction *involving* the query gene.
        # With it false BioGRID returns only pairs where both partners are in the list,
        # which for a single-gene query is just self-interactions - a first run reported
        # zero partners for every protein, including YDJ1, which actually has 1,010.
        url = (
            f"{BIOGRID_URL}?accessKey={key}&format=json&geneList={query}"
            f"&searchNames=true&includeInteractors=true&max={MAX_ROWS_PER_BATCH}"
        )
        async with semaphore:
            merged: dict[str, Any] = {}
            for page in range(MAX_PAGES):
                paged = f"{url}&start={page * MAX_ROWS_PER_BATCH}"
                try:
                    payload = await get_with_retry(session, paged)
                except (aiohttp.ClientError, RuntimeError) as exc:
                    for identifier in batch:
                        store.failures[identifier] = type(exc).__name__
                    return
                rows = payload if isinstance(payload, dict) else {}
                merged.update(rows)
                # A short page is the last page; BioGRID signals the end by returning
                # fewer rows than requested rather than by any explicit marker.
                if len(rows) < MAX_ROWS_PER_BATCH:
                    break
        for identifier in batch:
            store.records[identifier] = record_from_interactions(identifier, merged)

    await asyncio.gather(*(worker(batch) for batch in batches))
    return store


def write_interaction_store(store: InteractionStore, path: Path) -> None:
    """Write the store as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n", encoding="utf-8")
