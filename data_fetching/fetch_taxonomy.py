"""Superkingdom and lineage for the organisms in the store, from UniProt.

The layout table carries an organism name and nothing else, which is enough to block folds by
genus but not enough to ask the question the partner result now turns on: whether domain
architecture predicts which Hsp70 a protein binds *within a single kingdom*.

That distinction decides what the partition result means. Across kingdoms the answer is
trivially yes and tells you nothing - a bacterial protein partners DnaK because DnaK is the
Hsp70 its genome encodes, and its domain architecture is bacterial for the same phylogenetic
reason. Only inside a kingdom, where several Hsp70 paralogues coexist in one cell, can a
grouping be said to predict partner *choice*.

Inferring kingdom from the partner names would answer the question with itself: DnaK is the
bacterial answer, HSPA8 the eukaryotic one. So it is fetched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from data_fetching.utils import get_with_retry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

TAXONOMY_URL = "https://rest.uniprot.org/taxonomy/search?{query}"

# The three domains of life, plus the catch-all UniProt uses for sequences whose source is
# unresolved. Viruses are kept separate rather than folded into "other": a viral J-domain
# protein partners its host's Hsp70, which is a different situation again.
BACTERIA = "Bacteria"
ARCHAEA = "Archaea"
EUKARYOTA = "Eukaryota"
VIRUSES = "Viruses"
UNKNOWN = "unknown"

SUPERKINGDOMS = (BACTERIA, ARCHAEA, EUKARYOTA, VIRUSES)

DEFAULT_CONCURRENCY = 8
DEFAULT_CHECKPOINT_EVERY = 500


@dataclass
class TaxonomyStore:
    """Superkingdom per organism name, and the names that could not be resolved."""

    superkingdom: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, list[str]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Organisms resolved."""
        return len(self.superkingdom)

    def counts(self) -> dict[str, int]:
        """How many organisms fell in each superkingdom."""
        out = dict.fromkeys((*SUPERKINGDOMS, UNKNOWN), 0)
        for value in self.superkingdom.values():
            out[value] = out.get(value, 0) + 1
        return out

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the whole store."""
        return {
            "superkingdom": dict(self.superkingdom),
            "lineage": {k: list(v) for k, v in self.lineage.items()},
            "failures": dict(self.failures),
        }


def superkingdom_from_payload(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Read the superkingdom and lineage out of a UniProt taxonomy response.

    UniProt returns the lineage from the immediate parent upward, so the superkingdom is
    whichever of the three domains appears in it. Matching on the known set rather than
    taking the last element, because the lineage's top element is "cellular organisms" for
    everything cellular and would be useless.
    """
    results = payload.get("results") or []
    if not results:
        return UNKNOWN, []
    entry = results[0]
    names = [str(item.get("scientificName", "")) for item in entry.get("lineage", [])]
    # The organism's own rank counts too: a query for "Bacteria" resolves to itself.
    if entry.get("scientificName"):
        names.append(str(entry["scientificName"]))
    for candidate in SUPERKINGDOMS:
        if candidate in names:
            return candidate, names
    return UNKNOWN, names


def _load_checkpoint(path: Path | None) -> TaxonomyStore:
    """Reload a partial run, or start empty."""
    store = TaxonomyStore()
    if path is None or not path.exists():
        return store
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("checkpoint at %s is unreadable; starting fresh.", path, exc_info=True)
        return store
    store.superkingdom.update(payload.get("superkingdom", {}))
    store.lineage.update(payload.get("lineage", {}))
    store.failures.update(payload.get("failures", {}))
    return store


def _write_checkpoint(path: Path | None, store: TaxonomyStore) -> None:
    """Write the checkpoint atomically."""
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(store.to_json_dict()) + "\n")
    temporary.replace(path)


async def fetch_taxonomy(
    session: aiohttp.ClientSession,
    organisms: Sequence[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> TaxonomyStore:
    """Resolve each organism name to a superkingdom.

    Args:
        session: An open client session.
        organisms: Distinct organism names, as they appear in the layout table.
        concurrency: Maximum simultaneous requests.
        checkpoint: Where to write partial results and resume from.
        checkpoint_every: Organisms per batch between checkpoint writes.

    Returns:
        The store, including anything inherited from an existing checkpoint. An organism that
        resolves to nothing is recorded as ``unknown`` rather than omitted, so a resumed run
        does not retry it forever.
    """
    store = await asyncio.to_thread(_load_checkpoint, checkpoint)
    already = set(store.superkingdom) | set(store.failures)
    remaining = [name for name in organisms if name and name not in already]
    if already:
        logger.info("resuming: %d of %d organisms already resolved", len(already), len(organisms))

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def worker(name: str) -> None:
        # Strain suffixes ("Colletotrichum fioriniae PJ7") often miss; the binomial resolves.
        binomial = " ".join(name.split()[:2])
        query = urllib.parse.urlencode({"query": binomial, "fields": "lineage", "size": 1})
        async with semaphore:
            try:
                payload = await get_with_retry(session, TAXONOMY_URL.format(query=query))
            except (aiohttp.ClientError, RuntimeError) as exc:
                store.failures[name] = type(exc).__name__
                return
            kingdom, lineage = superkingdom_from_payload(payload)
            store.superkingdom[name] = kingdom
            store.lineage[name] = lineage

    for start in range(0, len(remaining), checkpoint_every):
        batch = remaining[start : start + checkpoint_every]
        await asyncio.gather(*(worker(name) for name in batch))
        await asyncio.to_thread(_write_checkpoint, checkpoint, store)
        logger.info("resolved %d of %d; %s", len(store), len(organisms), store.counts())

    return store


def write_taxonomy_store(store: TaxonomyStore, path: Path) -> None:
    """Write the finished store."""
    path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n")
