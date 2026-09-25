"""Interaction partners from STRING, for the organisms curated databases never reach.

BioGRID supplies partners for 1,089 of 217,701 J-domain proteins, and after requiring an
unambiguous Hsp70 partner that becomes 677. Every partner-based result in this project rests
on those 677, which is the binding constraint on all of them - no statistical care raises it.
The shortfall is not a fetching failure: BioGRID curates the literature, and the literature
is yeast, human, fly, and worm.

STRING covers thousands of genomes, most of them bacterial, and a probe of 40 proteins with
no BioGRID record found partners for 28% of them - projecting to roughly 60,000. That is
where DnaJ and DnaK actually live. Most J-domain proteins in this dataset are bacterial, the
DnaJ-DnaK pair is the ancestral form of the system, and until now none of them contributed a
single partner observation.

What the scores mean, and why they are kept separate
----------------------------------------------------
STRING reports one score per evidence channel, and they are emphatically not the same kind
of evidence:

``experimental`` (escore) and ``database`` (dscore)
    An experiment, or a curator reading one. For a bacterium these are usually *transferred*
    from a better-studied relative by orthology - which is inference, but inference from
    whole-protein orthology rather than from domain architecture. That distinction is what
    keeps it usable here: a partner assignment derived from domain content could not then
    test whether domain content predicts partners.
``neighborhood`` (nscore), ``fusion`` (fscore), ``cooccurrence`` (pscore)
    Genomic context. Genes that sit together, fuse, or appear and vanish together across
    genomes. Available for any sequenced genome, and independent of both protein structure
    and curated interaction data - the most genuinely orthogonal evidence available at this
    scale.
``coexpression`` (ascore), ``textmining`` (tscore)
    Kept but not used by default. Text mining in particular reflects what has been written
    about a protein, which tracks how well studied its organism is - the exact bias that
    made BioGRID unusable for this question.

Channels are stored separately rather than combined, so an analysis states which evidence it
rests on and a result computed on one channel is never silently pooled with another.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from data_fetching.utils import RATE_LIMIT_STATUSES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

STRING_URL = "https://string-db.org/api/json/interaction_partners?{query}"

# Per-protein partner cap. Fifty is deep enough to include the chaperones for a J-domain
# protein and shallow enough that one hub does not dominate the store.
DEFAULT_LIMIT = 50

# STRING asks callers to identify themselves and rate-limits accordingly.
CALLER_IDENTITY = "jhu_bioreu_jdp"

# STRING scores are 0 to 1; this is the floor for a channel to count as supporting evidence.
# 0.15 is STRING's own "low confidence" cut, kept low here because the question is which
# partners exist at all, with the strength recorded for the caller to threshold again.
MIN_CHANNEL_SCORE = 0.15

NOT_FOUND = 404

# Attempts before giving up on a rate-limited request.
MAX_ATTEMPTS = 5

DEFAULT_CONCURRENCY = 8
DEFAULT_CHECKPOINT_EVERY = 2000


@dataclass(frozen=True)
class FetchSettings:
    """How hard to pull, how often to checkpoint, and how deep to go per protein."""

    concurrency: int = DEFAULT_CONCURRENCY
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    limit: int = DEFAULT_LIMIT


DEFAULT_FETCH = FetchSettings()

# Channel names as STRING returns them, mapped to what they actually mean.
CHANNEL_EXPERIMENTAL = "experimental"
CHANNEL_DATABASE = "database"
CHANNEL_NEIGHBORHOOD = "neighborhood"
CHANNEL_FUSION = "fusion"
CHANNEL_COOCCURRENCE = "cooccurrence"
CHANNEL_COEXPRESSION = "coexpression"
CHANNEL_TEXTMINING = "textmining"

_SCORE_KEYS = {
    CHANNEL_EXPERIMENTAL: "escore",
    CHANNEL_DATABASE: "dscore",
    CHANNEL_NEIGHBORHOOD: "nscore",
    CHANNEL_FUSION: "fscore",
    CHANNEL_COOCCURRENCE: "pscore",
    CHANNEL_COEXPRESSION: "ascore",
    CHANNEL_TEXTMINING: "tscore",
}

# Evidence that some experiment, somewhere, supports the pair.
CURATED_CHANNELS = frozenset({CHANNEL_EXPERIMENTAL, CHANNEL_DATABASE})

# Evidence from genome organisation, independent of structure and of curation.
CONTEXT_CHANNELS = frozenset({CHANNEL_NEIGHBORHOOD, CHANNEL_FUSION, CHANNEL_COOCCURRENCE})

# STRING answers 404 for an accession it has never heard of, which for this dataset is the
# common case rather than an error: most J-domain proteins here come from genomes STRING has
# not assembled a network for. Recorded as a distinct outcome so the final accounting can
# separate "STRING has no entry for this protein" from "the request failed", and so a
# resumed run does not retry either.
NOT_IN_STRING = "not_in_string"


@dataclass(frozen=True)
class Partner:
    """One interaction partner, with the evidence supporting it kept per channel."""

    name: str
    scores: dict[str, float] = field(default_factory=dict)

    def supported_by(self, channels: frozenset[str], *, minimum: float = MIN_CHANNEL_SCORE) -> bool:
        """Whether any of the named channels supports this pair above the floor."""
        return any(self.scores.get(channel, 0.0) >= minimum for channel in channels)

    def to_json_dict(self) -> dict[str, object]:
        """Serialise, dropping channels that contributed nothing."""
        return {"name": self.name, "scores": {k: round(v, 4) for k, v in self.scores.items() if v > 0}}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> Partner:
        """Rebuild from a checkpoint."""
        return cls(name=payload["name"], scores=dict(payload.get("scores", {})))


@dataclass
class StringStore:
    """Partners per protein, and the accessions that returned nothing or failed."""

    partners: dict[str, list[Partner]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of proteins with at least one partner."""
        return len(self.partners)

    @property
    def n_absent(self) -> int:
        """Accessions STRING has no entry for, as opposed to requests that failed."""
        return sum(1 for reason in self.failures.values() if reason == NOT_IN_STRING)

    @property
    def n_failed(self) -> int:
        """Requests that genuinely failed and could be worth retrying."""
        return sum(1 for reason in self.failures.values() if reason != NOT_IN_STRING)

    def names_by_accession(
        self,
        channels: frozenset[str],
        *,
        minimum: float = MIN_CHANNEL_SCORE,
    ) -> dict[str, list[str]]:
        """Partner names per protein, restricted to one kind of evidence.

        The channel set is required rather than defaulted, because "which partners does this
        protein have" has a different answer for curated evidence and for genomic context,
        and a caller that has not chosen between them has not yet asked a well-posed
        question.
        """
        out: dict[str, list[str]] = {}
        for accession, partners in self.partners.items():
            names = [p.name for p in partners if p.supported_by(channels, minimum=minimum)]
            if names:
                out[accession] = names
        return out

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the whole store."""
        return {
            "partners": {k: [p.to_json_dict() for p in v] for k, v in self.partners.items()},
            "failures": dict(self.failures),
        }


async def _get_json(session: aiohttp.ClientSession, url: str) -> object:
    """Fetch one STRING response, tolerating how it labels its own JSON.

    The shared ``get_with_retry`` cannot be used here. It calls ``resp.json()``, which
    enforces a JSON content type, and STRING serves its JSON endpoint under a different
    one - so every successful response raised ``ContentTypeError`` and was discarded as a
    failure. On the first attempt that silently threw away 1,508 of 2,000 responses, all of
    them real data, and reported the run as 75% failed.

    Raises:
        FileNotFoundError: STRING has no entry for this identifier.
        RuntimeError: Rate-limited past ``MAX_ATTEMPTS``, or an unexpected status.
    """
    for attempt in range(MAX_ATTEMPTS):
        async with session.get(url) as response:
            if response.status == NOT_FOUND:
                message = "not in STRING"
                raise FileNotFoundError(message)
            if response.status in RATE_LIMIT_STATUSES:
                await asyncio.sleep(min(60 * (2**attempt), 300))
                continue
            response.raise_for_status()
            # content_type=None: parse the body as JSON whatever the header claims it is.
            return await response.json(content_type=None)
    message = f"rate limited past {MAX_ATTEMPTS} attempts"
    raise RuntimeError(message)


def partners_from_payload(payload: object) -> list[Partner]:
    """Read STRING's interaction_partners response into per-channel scores."""
    if not isinstance(payload, list):
        return []
    partners: list[Partner] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("preferredName_B") or row.get("stringId_B")
        if not name:
            continue
        scores: dict[str, float] = {}
        for channel, key in _SCORE_KEYS.items():
            try:
                value = float(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                scores[channel] = value
        partners.append(Partner(name=str(name), scores=scores))
    return partners


def _load_checkpoint(path: Path | None) -> StringStore:
    """Reload a partial run, or start empty."""
    store = StringStore()
    if path is None or not path.exists():
        return store
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("checkpoint at %s is unreadable; starting fresh.", path, exc_info=True)
        return store
    for accession, rows in payload.get("partners", {}).items():
        store.partners[accession] = [Partner.from_json_dict(row) for row in rows]
    store.failures.update(payload.get("failures", {}))
    return store


def _write_checkpoint(path: Path | None, store: StringStore) -> None:
    """Write the checkpoint atomically, so a kill mid-write cannot corrupt it."""
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(store.to_json_dict()) + "\n")
    temporary.replace(path)


async def fetch_string_partners(
    session: aiohttp.ClientSession,
    accessions: Sequence[str],
    *,
    checkpoint: Path | None = None,
    settings: FetchSettings = DEFAULT_FETCH,
) -> StringStore:
    """Fetch interaction partners for every accession.

    Concurrency is deliberately lower than the other fetchers here: STRING is a public
    service that asks callers to identify themselves, and hammering it would be both rude
    and self-defeating.

    Args:
        session: An open client session.
        accessions: Accessions to fetch.
        checkpoint: Where to write partial results and resume from.
        settings: Concurrency, checkpoint interval, and partners requested per protein.

    Returns:
        The store, including anything inherited from an existing checkpoint. An accession
        with no partners is recorded as fetched with an empty list rather than omitted, so a
        resumed run does not retry it forever.
    """
    store = await asyncio.to_thread(_load_checkpoint, checkpoint)
    already = set(store.partners) | set(store.failures)
    remaining = [accession for accession in accessions if accession not in already]
    if already:
        logger.info("resuming: %d of %d accessions already fetched", len(already), len(accessions))

    semaphore = asyncio.Semaphore(max(1, settings.concurrency))

    async def worker(accession: str) -> None:
        query = urllib.parse.urlencode(
            {"identifiers": accession, "limit": settings.limit, "caller_identity": CALLER_IDENTITY}
        )
        async with semaphore:
            try:
                payload = await _get_json(session, STRING_URL.format(query=query))
            except FileNotFoundError:
                # STRING has no entry for this protein: an answer, not a fault.
                store.failures[accession] = NOT_IN_STRING
                return
            except (aiohttp.ClientError, RuntimeError, ValueError) as exc:
                store.failures[accession] = type(exc).__name__
                return
            store.partners[accession] = partners_from_payload(payload)

    for start in range(0, len(remaining), settings.checkpoint_every):
        batch = remaining[start : start + settings.checkpoint_every]
        await asyncio.gather(*(worker(accession) for accession in batch))
        await asyncio.to_thread(_write_checkpoint, checkpoint, store)
        with_partners = sum(1 for rows in store.partners.values() if rows)
        logger.info(
            "fetched %d of %d; %d carry partners, %d absent from STRING, %d failed",
            len(store.partners) + len(store.failures),
            len(accessions),
            with_partners,
            store.n_absent,
            store.n_failed,
        )

    return store


def write_string_store(store: StringStore, path: Path) -> None:
    """Write the finished store."""
    path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n")
