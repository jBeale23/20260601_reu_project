"""Gene Ontology annotation with its evidence code, from QuickGO.

``fetch_function`` reads GO terms out of each protein's UniProtKB entry, and that route has a
hard ceiling: UniProt exposes evidence records only for *reviewed* entries, and 220 of the
217,701 J-domain proteins here are reviewed. It therefore finds 198 proteins with
experimentally-backed terms, which is too few to support any analysis that needs both
function and a feature profile.

QuickGO serves the same annotations from the GO consortium's own store, including
annotations contributed by model-organism databases that never reach a UniProtKB evidence
record. Sampling 400 of our accessions against it found roughly 2.7x more experimental
annotation than the UniProt route, plus a substantially larger pool of annotations that are
manual or phylogenetic rather than electronic.

Evidence tiers
--------------
Terms are kept in tiers rather than pooled, because they are not the same kind of evidence
and the difference decides what a result means:

``experimental``
    Somebody did an experiment on this protein. The only tier that can test a hypothesis
    about domain architecture without assuming it.
``phylogenetic``
    Propagated to this protein from an experimentally annotated relative through a
    phylogenetic tree (IBA and relatives, from the GO Reference Genome project). Grounded in
    experiment at one remove. Trees are built from whole-sequence phylogeny rather than
    domain signatures, so this is less circular with a domain-architecture claim than the
    electronic route is - but it is still inference, not observation.
``curated``
    A curator's manual assertion: sequence similarity, an author's statement, or curator
    inference.
``electronic``
    Assigned by an automatic pipeline, overwhelmingly InterPro2GO and PANTHER TreeGrafter.
    Excluded from every tier above because the commonest route is "this domain signature
    implies this term" - using it to test whether domain architecture predicts function
    assumes the answer.

A protein appears in a tier only if it carries at least one term at that level of evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from data_fetching.utils import get_with_retry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={accession}&limit=100"

# GO evidence codes, grouped by what they actually assert.
EXPERIMENTAL_CODES = frozenset({"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"})
PHYLOGENETIC_CODES = frozenset({"IBA", "IBD", "IKR", "IRD"})
CURATED_CODES = frozenset({"TAS", "NAS", "IC", "ISS", "ISO", "ISA", "ISM", "IGC", "RCA"})
ELECTRONIC_CODES = frozenset({"IEA"})

# "No biological data available" - an explicit statement that nothing is known, which must
# not be read as an annotation.
NO_DATA_CODES = frozenset({"ND"})

TIER_EXPERIMENTAL = "experimental"
TIER_PHYLOGENETIC = "phylogenetic"
TIER_CURATED = "curated"
TIER_ELECTRONIC = "electronic"

DEFAULT_CONCURRENCY = 16
DEFAULT_CHECKPOINT_EVERY = 2000


def tier_for(code: str | None) -> str | None:
    """Which evidence tier a GO evidence code belongs to, or ``None`` if it asserts nothing."""
    if not code or code in NO_DATA_CODES:
        return None
    if code in EXPERIMENTAL_CODES:
        return TIER_EXPERIMENTAL
    if code in PHYLOGENETIC_CODES:
        return TIER_PHYLOGENETIC
    if code in CURATED_CODES:
        return TIER_CURATED
    if code in ELECTRONIC_CODES:
        return TIER_ELECTRONIC
    return None


@dataclass
class EvidenceRecord:
    """One protein's GO terms, separated by how well supported each is."""

    accession: str
    experimental: tuple[str, ...] = ()
    phylogenetic: tuple[str, ...] = ()
    curated: tuple[str, ...] = ()
    electronic: tuple[str, ...] = ()

    def terms_for(self, tier: str) -> tuple[str, ...]:
        """Terms at one tier."""
        return {
            TIER_EXPERIMENTAL: self.experimental,
            TIER_PHYLOGENETIC: self.phylogenetic,
            TIER_CURATED: self.curated,
            TIER_ELECTRONIC: self.electronic,
        }.get(tier, ())

    def terms_at_or_above(self, tier: str) -> tuple[str, ...]:
        """Terms at the given tier and every better-supported one.

        Ordered experimental, phylogenetic, curated, electronic. Asking for ``curated``
        therefore includes experimental and phylogenetic terms too, which is what a caller
        wanting "everything not purely electronic" means.
        """
        order = [TIER_EXPERIMENTAL, TIER_PHYLOGENETIC, TIER_CURATED, TIER_ELECTRONIC]
        if tier not in order:
            return ()
        collected: list[str] = []
        for level in order[: order.index(tier) + 1]:
            collected.extend(self.terms_for(level))
        return tuple(dict.fromkeys(collected))

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the checkpoint and the store."""
        return {
            "accession": self.accession,
            "experimental": list(self.experimental),
            "phylogenetic": list(self.phylogenetic),
            "curated": list(self.curated),
            "electronic": list(self.electronic),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> EvidenceRecord:
        """Rebuild from a checkpoint."""
        return cls(
            accession=payload["accession"],
            experimental=tuple(payload.get("experimental", ())),
            phylogenetic=tuple(payload.get("phylogenetic", ())),
            curated=tuple(payload.get("curated", ())),
            electronic=tuple(payload.get("electronic", ())),
        )


@dataclass
class EvidenceStore:
    """Every fetched record, and every accession that could not be fetched."""

    records: dict[str, EvidenceRecord] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of fetched records."""
        return len(self.records)

    def tier_counts(self) -> dict[str, int]:
        """How many proteins carry at least one term at each tier."""
        return {
            tier: sum(1 for record in self.records.values() if record.terms_for(tier))
            for tier in (TIER_EXPERIMENTAL, TIER_PHYLOGENETIC, TIER_CURATED, TIER_ELECTRONIC)
        }

    def terms_by_accession(self, tier: str, *, cumulative: bool = False) -> dict[str, tuple[str, ...]]:
        """Terms per protein at one tier, for proteins that have any.

        Args:
            tier: Which tier to read.
            cumulative: Include better-supported tiers as well.
        """
        out: dict[str, tuple[str, ...]] = {}
        for accession, record in self.records.items():
            terms = record.terms_at_or_above(tier) if cumulative else record.terms_for(tier)
            if terms:
                out[accession] = terms
        return out

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the whole store."""
        return {
            "records": {key: value.to_json_dict() for key, value in self.records.items()},
            "failures": dict(self.failures),
        }


def record_from_payload(accession: str, payload: dict[str, Any]) -> EvidenceRecord:
    """Sort one protein's QuickGO annotations into evidence tiers.

    A term is filed at the best evidence that supports it. The same term is commonly
    annotated several times from different sources - once experimentally and three times
    electronically - and filing it at every tier would make the electronic tier look like it
    contained experimental knowledge.
    """
    best: dict[str, str] = {}
    rank = {TIER_EXPERIMENTAL: 0, TIER_PHYLOGENETIC: 1, TIER_CURATED: 2, TIER_ELECTRONIC: 3}
    for row in payload.get("results", []):
        tier = tier_for(row.get("goEvidence"))
        if tier is None:
            continue
        # Prefer the term name; fall back to the id when QuickGO omits the name.
        term = row.get("goName") or row.get("goId")
        if not term:
            continue
        current = best.get(term)
        if current is None or rank[tier] < rank[current]:
            best[term] = tier

    buckets: dict[str, list[str]] = {tier: [] for tier in rank}
    for term, tier in best.items():
        buckets[tier].append(term)
    return EvidenceRecord(
        accession=accession,
        experimental=tuple(sorted(buckets[TIER_EXPERIMENTAL])),
        phylogenetic=tuple(sorted(buckets[TIER_PHYLOGENETIC])),
        curated=tuple(sorted(buckets[TIER_CURATED])),
        electronic=tuple(sorted(buckets[TIER_ELECTRONIC])),
    )


def _load_checkpoint(path: Path | None) -> EvidenceStore:
    """Reload a partial run, or start empty."""
    store = EvidenceStore()
    if path is None or not path.exists():
        return store
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("checkpoint at %s is unreadable; starting fresh.", path, exc_info=True)
        return store
    for accession, record in payload.get("records", {}).items():
        store.records[accession] = EvidenceRecord.from_json_dict(record)
    store.failures.update(payload.get("failures", {}))
    return store


def _write_checkpoint(path: Path | None, store: EvidenceStore) -> None:
    """Write the checkpoint atomically, so a kill mid-write cannot corrupt it."""
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(store.to_json_dict()) + "\n")
    temporary.replace(path)


async def fetch_go_evidence(
    session: aiohttp.ClientSession,
    accessions: Sequence[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> EvidenceStore:
    """Fetch evidence-coded GO annotation for every accession.

    Checkpointed for the same reason ``fetch_functions`` is: this is 217,701 requests
    against a rate-limited public API, and an un-checkpointed run that dies at hour eleven
    has produced nothing.

    Args:
        session: An open client session.
        accessions: Accessions to fetch.
        concurrency: Maximum simultaneous requests.
        checkpoint: Where to write partial results and resume from.
        checkpoint_every: Accessions per batch between checkpoint writes.

    Returns:
        The store, including anything inherited from an existing checkpoint.
    """
    store = await asyncio.to_thread(_load_checkpoint, checkpoint)
    already = set(store.records) | set(store.failures)
    remaining = [accession for accession in accessions if accession not in already]
    if already:
        logger.info("resuming: %d of %d accessions already fetched", len(already), len(accessions))

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def worker(accession: str) -> None:
        async with semaphore:
            try:
                payload = await get_with_retry(session, QUICKGO_URL.format(accession=accession))
            except (aiohttp.ClientError, RuntimeError) as exc:
                store.failures[accession] = type(exc).__name__
                return
            store.records[accession] = record_from_payload(accession, payload)

    for start in range(0, len(remaining), checkpoint_every):
        batch = remaining[start : start + checkpoint_every]
        await asyncio.gather(*(worker(accession) for accession in batch))
        await asyncio.to_thread(_write_checkpoint, checkpoint, store)
        logger.info(
            "fetched %d of %d; tiers %s",
            len(store.records),
            len(accessions),
            store.tier_counts(),
        )

    return store


def write_evidence_store(store: EvidenceStore, path: Path) -> None:
    """Write the finished store."""
    path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n")
