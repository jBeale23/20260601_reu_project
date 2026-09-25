"""Fetch functional annotation from UniProt for a set of accessions.

Structural difference is not functional difference. The project has shown that the two
largest J-domain protein classes each split into structurally distinct populations; whether
those populations *do* different things is a separate question, and it is the one that
decides whether the split describes new classes or merely new shapes.

UniProt carries four kinds of evidence that bear on it, none of them derived from the
domain architecture the classes are defined by:

- **Gene Ontology terms**, split into molecular function, biological process, and cellular
  component. Molecular function is the most direct: "Hsp70 protein binding" and "ATPase
  activator activity" are claims about what the protein does.
- **Subcellular location**, which is functional in the sense that matters here - a
  mitochondrial outer-membrane JDP and a nucleolar one cannot be doing the same job.
- **Keywords**, a controlled vocabulary that is coarser than GO but far better populated.
- **Curated binary interactions**, which name actual partners.

The coverage caveat, measured rather than assumed
-------------------------------------------------
Functional annotation is concentrated in model organisms for the same reason the class
labels are, so a naive enrichment test would mostly rediscover which proteins are
well-studied. Coverage is therefore reported per class and per subclass, and any
enrichment test is run *within* the annotated subset rather than across the annotated and
unannotated together.
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
    from collections.abc import Iterator, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

# UniProt asks for a modest request rate from unauthenticated clients; this matches the
# pacing used for the InterPro fetches, which ran to completion without a single failure.
# Accessions per checkpoint batch. Small enough that a kill costs minutes, large enough
# that the checkpoint write is not itself the bottleneck.
DEFAULT_CHECKPOINT_EVERY = 2000
DEFAULT_CONCURRENCY = 6
DEFAULT_RATE_LIMIT = 8.0

# GO aspect codes as they appear in the cross-reference term strings.
ASPECT_FUNCTION = "F"
ASPECT_PROCESS = "P"
ASPECT_COMPONENT = "C"

# Evidence Ontology codes for terms supported by an actual experiment: direct assay,
# physical interaction, mutant phenotype, genetic interaction, expression pattern, and
# high-throughput variants of the same.
#
# The distinction is load-bearing rather than pedantic. A term without one of these is
# usually inferred electronically, and the commonest route for that inference is
# InterPro2GO - the domain signature implies the term. Using such a term to test whether
# domain architecture predicts function would be circular: the answer was assumed when the
# annotation was made. A probe of 400 proteins found 74% carrying GO terms but only 0.8%
# reviewed, and an unreviewed entry carried zero evidence records at all.
EXPERIMENTAL_EVIDENCE = frozenset(
    {
        "ECO:0000314",  # direct assay
        "ECO:0000353",  # physical interaction
        "ECO:0000315",  # mutant phenotype
        "ECO:0000316",  # genetic interaction
        "ECO:0000270",  # expression pattern
        "ECO:0000269",  # experimental evidence used in manual assertion
        "ECO:0007005",  # high-throughput direct assay
        "ECO:0007001",  # high-throughput mutant phenotype
        "ECO:0007003",  # high-throughput genetic interaction
    }
)

# A GO term string is "<aspect>:<name>", so anything shorter carries no name.
_MIN_GO_TERM_LENGTH = 2


@dataclass(frozen=True, slots=True)
class FunctionalRecord:
    """Functional annotation for one protein."""

    accession: str
    go_function: tuple[str, ...] = ()
    go_process: tuple[str, ...] = ()
    go_component: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    subcellular_locations: tuple[str, ...] = ()
    n_interactions: int = 0
    is_reviewed: bool = False
    # Terms backed by an experiment rather than inferred electronically. Kept separate
    # because only these can test a hypothesis about domain architecture without
    # circularity.
    experimental_terms: tuple[str, ...] = ()
    # Curated disease associations, from DISEASE comments rather than disease-sounding
    # keywords: a keyword says only that some variant is pathogenic, while the comment
    # names the disease and cross-references OMIM, which is what makes it groupable. Only
    # reviewed entries carry these.
    diseases: tuple[str, ...] = ()
    disease_acronyms: tuple[str, ...] = ()
    disease_mim_ids: tuple[str, ...] = ()

    @property
    def has_disease_association(self) -> bool:
        """Whether a curated disease is linked to this protein."""
        return bool(self.diseases)

    @property
    def has_experimental_annotation(self) -> bool:
        """Whether any term is backed by an experiment rather than electronic inference."""
        return bool(self.experimental_terms)

    @property
    def has_any_annotation(self) -> bool:
        """Whether this protein carries functional evidence of any kind."""
        return bool(self.go_function or self.go_process or self.go_component or self.subcellular_locations)

    @property
    def n_go_terms(self) -> int:
        """Total Gene Ontology terms across all three aspects."""
        return len(self.go_function) + len(self.go_process) + len(self.go_component)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for the store."""
        return {
            "accession": self.accession,
            "go_function": list(self.go_function),
            "go_process": list(self.go_process),
            "go_component": list(self.go_component),
            "keywords": list(self.keywords),
            "subcellular_locations": list(self.subcellular_locations),
            "n_interactions": self.n_interactions,
            "is_reviewed": self.is_reviewed,
            "experimental_terms": list(self.experimental_terms),
            "diseases": list(self.diseases),
            "disease_acronyms": list(self.disease_acronyms),
            "disease_mim_ids": list(self.disease_mim_ids),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> FunctionalRecord:
        """Rebuild from stored JSON."""
        return cls(
            accession=str(payload["accession"]),
            go_function=tuple(payload.get("go_function", [])),
            go_process=tuple(payload.get("go_process", [])),
            go_component=tuple(payload.get("go_component", [])),
            keywords=tuple(payload.get("keywords", [])),
            subcellular_locations=tuple(payload.get("subcellular_locations", [])),
            n_interactions=int(payload.get("n_interactions", 0)),
            is_reviewed=bool(payload.get("is_reviewed", False)),
            experimental_terms=tuple(payload.get("experimental_terms", [])),
            # Defaulted, so a store written before disease capture still loads. A run
            # against the older format simply reports no disease associations rather than
            # failing, which matters because a 21-hour fetch was already in flight.
            diseases=tuple(payload.get("diseases", [])),
            disease_acronyms=tuple(payload.get("disease_acronyms", [])),
            disease_mim_ids=tuple(payload.get("disease_mim_ids", [])),
        )


@dataclass(slots=True)
class FunctionStore:
    """Every functional record fetched, keyed by accession."""

    records: dict[str, FunctionalRecord] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of records held."""
        return len(self.records)

    def __iter__(self) -> Iterator[FunctionalRecord]:
        """Iterate records in accession order."""
        return iter([self.records[key] for key in sorted(self.records)])

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the whole store."""
        return {
            "records": {key: value.to_json_dict() for key, value in self.records.items()},
            "failures": dict(self.failures),
        }


def _go_terms(payload: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    """Gene Ontology terms by aspect, plus the subset backed by an experiment."""
    aspects: dict[str, list[str]] = {ASPECT_FUNCTION: [], ASPECT_PROCESS: [], ASPECT_COMPONENT: []}
    experimental: list[str] = []
    for reference in payload.get("uniProtKBCrossReferences", []):
        if reference.get("database") != "GO":
            continue
        backed = any(item.get("evidenceCode") in EXPERIMENTAL_EVIDENCE for item in reference.get("evidences", []))
        for prop in reference.get("properties", []):
            # Term strings look like "F:Hsp70 protein binding"; the prefix is the aspect.
            if prop.get("key") == "GoTerm" and len(prop.get("value", "")) > _MIN_GO_TERM_LENGTH:
                value = prop["value"]
                aspects.setdefault(value[0], []).append(value[2:])
                if backed:
                    experimental.append(value[2:])
    return aspects, experimental


def _locations_and_interactions(payload: dict[str, Any]) -> tuple[list[str], int]:
    """Subcellular locations and the count of curated binary interactions."""
    locations: list[str] = []
    interactions = 0
    for comment in payload.get("comments", []):
        kind = comment.get("commentType")
        if kind == "SUBCELLULAR LOCATION":
            for item in comment.get("subcellularLocations", []):
                value = (item.get("location") or {}).get("value")
                if value:
                    locations.append(value)
        elif kind == "INTERACTION":
            interactions += len(comment.get("interactions", []))
    return locations, interactions


def _diseases(payload: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Curated disease associations: names, acronyms, and OMIM ids.

    Taken from ``DISEASE`` comments rather than from disease-sounding keywords. A keyword
    such as "Disease variant" says only that some variant is pathogenic; the comment names
    the disease and cross-references OMIM, which is what makes it groupable.
    """
    names: list[str] = []
    acronyms: list[str] = []
    mim_ids: list[str] = []
    for comment in payload.get("comments", []):
        if comment.get("commentType") != "DISEASE":
            continue
        disease = comment.get("disease") or {}
        name = (disease.get("diseaseId") or "").strip()
        acronym = (disease.get("acronym") or "").strip()
        reference = disease.get("diseaseCrossReference") or {}
        if name:
            names.append(name)
        if acronym:
            acronyms.append(acronym)
        if reference.get("database") == "MIM" and reference.get("id"):
            mim_ids.append(str(reference["id"]))
    return tuple(names), tuple(acronyms), tuple(mim_ids)


def record_from_payload(accession: str, payload: dict[str, Any]) -> FunctionalRecord:
    """Extract the functional fields from one UniProt entry."""
    aspects, experimental = _go_terms(payload)
    locations, interactions = _locations_and_interactions(payload)

    diseases, disease_acronyms, disease_mim_ids = _diseases(payload)
    return FunctionalRecord(
        accession=accession,
        go_function=tuple(dict.fromkeys(aspects[ASPECT_FUNCTION])),
        go_process=tuple(dict.fromkeys(aspects[ASPECT_PROCESS])),
        go_component=tuple(dict.fromkeys(aspects[ASPECT_COMPONENT])),
        keywords=tuple(item["name"] for item in payload.get("keywords", []) if item.get("name")),
        subcellular_locations=tuple(dict.fromkeys(locations)),
        n_interactions=interactions,
        is_reviewed=payload.get("entryType", "").startswith("UniProtKB reviewed"),
        experimental_terms=tuple(dict.fromkeys(experimental)),
        diseases=diseases,
        disease_acronyms=disease_acronyms,
        disease_mim_ids=disease_mim_ids,
    )


def _load_checkpoint(path: Path | None) -> FunctionStore:
    """Load a checkpoint if one is there, otherwise start empty."""
    if path is not None and path.exists():
        return load_function_store(path)
    return FunctionStore()


def _write_checkpoint(store: FunctionStore, path: Path) -> None:
    """Write the partial store, atomically.

    Via a temporary file and :func:`os.replace` so a job killed mid-write leaves the
    previous checkpoint intact rather than a truncated one - the failure mode that would
    turn a crash into a total loss instead of a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def fetch_functions(
    session: aiohttp.ClientSession,
    accessions: Sequence[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> FunctionStore:
    """Fetch functional annotation for every accession.

    Checkpointing is not optional decoration at this size. A run over the full store is
    181,526 requests against a rate-limited API; the first attempt was killed by a
    twelve-hour wall clock having written nothing, because the store was only serialized
    after the last request returned, so twelve hours of fetching was discarded. Passing
    ``checkpoint`` makes the work survive that: completed accessions are reloaded and
    skipped, and progress is written every ``checkpoint_every`` accessions.

    Args:
        session: An open client session.
        accessions: Accessions to fetch, in any order.
        concurrency: Maximum simultaneous requests.
        checkpoint: Where to write partial results, and where to resume from. ``None``
            keeps the previous all-or-nothing behaviour.
        checkpoint_every: Accessions per batch; the checkpoint is written after each.

    Returns:
        The store, including anything inherited from an existing checkpoint.
    """
    # Off the event loop: these are blocking reads and writes, and the store runs to
    # tens of megabytes, which is long enough to stall every in-flight request.
    store = await asyncio.to_thread(_load_checkpoint, checkpoint)
    already_done = set(store.records) | set(store.failures)
    remaining = [accession for accession in accessions if accession not in already_done]
    if already_done:
        logger.info("resuming: %s of %s accessions already fetched", len(already_done), len(accessions))

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def worker(accession: str) -> None:
        async with semaphore:
            try:
                payload = await get_with_retry(session, UNIPROT_URL.format(accession=accession))
            except (aiohttp.ClientError, RuntimeError) as exc:
                store.failures[accession] = type(exc).__name__
                return
            store.records[accession] = record_from_payload(accession, payload)

    # Batched rather than one gather over every accession: it bounds how much work a kill
    # can destroy, and it stops 181,526 coroutines being built before the first request.
    batch_size = max(1, checkpoint_every)
    for start in range(0, len(remaining), batch_size):
        await asyncio.gather(*(worker(accession) for accession in remaining[start : start + batch_size]))
        if checkpoint is not None:
            await asyncio.to_thread(_write_checkpoint, store, checkpoint)
        logger.info(
            "fetched %s of %s (%s failures)",
            len(store.records) + len(store.failures),
            len(accessions),
            len(store.failures),
        )
    return store


def write_function_store(store: FunctionStore, path: Path) -> None:
    """Write the store as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def load_function_store(path: Path) -> FunctionStore:
    """Read a store written by :func:`write_function_store`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FunctionStore(
        records={key: FunctionalRecord.from_json_dict(value) for key, value in payload.get("records", {}).items()},
        failures=dict(payload.get("failures", {})),
    )
