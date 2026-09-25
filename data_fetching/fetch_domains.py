"""Fetch the complete InterPro domain annotation and sequence for each UniProt accession.

The architecture fetch (``fetch-architectures-dnaj``) returns only protein *metadata*:
no sequence and no per-protein domain coordinates. This module fills that gap by
querying, for every accession:

* ``entry/all/protein/uniprot/<accession>`` — every InterPro and member-database
  signature match with residue coordinates;
* ``protein/uniprot/<accession>`` — the amino-acid sequence and organism metadata.

Results are written as a *domain store* (see :mod:`domain_layout.records`), which is
the input for region routing, disorder prediction, SHARK scoring, and classification.

Output format::

    {
        "store_version": 1,
        "source": "ipr001623_domain_architectures_no_dedup.json",
        "n_proteins": 2,
        "n_failed": 0,
        "proteins": {"P08622": {"sequence": "MAKQ...", "entries": []}},
        "failures": {},
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from tqdm import tqdm

from data_fetching.utils import INTERPRO_HEADERS, configure_logging, get_with_retry
from domain_layout.records import (
    DomainEntry,
    DomainFragment,
    DomainStore,
    ProteinDomainRecord,
    load_domain_store,
    write_domain_store,
)
from scripts.extract_uniprot_ids import extract_accessions, fetch_warnings, normalize_accession

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

configure_logging()
logger = logging.getLogger(__name__)

ENTRIES_URL_TEMPLATE = (
    "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{accession}/?page_size=200&format=json"
)
PROTEIN_URL_TEMPLATE = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/{accession}/?format=json"

_DEFAULT_OUTPUT = Path("protein_domains.json")
_DEFAULT_CONCURRENCY = 5
_DEFAULT_CHECKPOINT_EVERY = 200
_DEFAULT_TIMEOUT_SECONDS = 300
# Steady request rate for the InterPro API. Several array tasks fetch at once, so this
# is deliberately conservative; raise it only if EBI confirms a higher rate is welcome.
_DEFAULT_REQUESTS_PER_SECOND = 10.0
_NOT_FOUND_STATUS = 404

FAILURE_NOT_FOUND = "not_found"
FAILURE_RETRIES_EXHAUSTED = "retries_exhausted"
FAILURE_EMPTY_RESPONSE = "empty_response"


class RateLimiter:
    """Paces requests to a steady rate shared by every concurrent worker.

    A full DnaJ fetch is hundreds of thousands of requests, and several SLURM array
    tasks run at once, so politeness has to be enforced per process rather than left to
    the concurrency limit. Each caller reserves the next free slot, which spreads
    requests evenly instead of releasing them in bursts.
    """

    def __init__(self, requests_per_second: float) -> None:
        """Create a limiter; a non-positive rate disables pacing."""
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    @property
    def enabled(self) -> bool:
        """Whether pacing is active."""
        return self._min_interval > 0.0

    async def acquire(self) -> None:
        """Wait until this caller's slot in the request schedule."""
        if not self.enabled:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_slot = max(now, self._next_slot) + self._min_interval


@dataclass(frozen=True, slots=True)
class DomainFetchOptions:
    """Tunables for a domain-store fetch run."""

    concurrency: int = _DEFAULT_CONCURRENCY
    include_sequence: bool = True
    checkpoint_path: Path | None = None
    checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY
    show_progress: bool = True
    requests_per_second: float = _DEFAULT_REQUESTS_PER_SECOND


DEFAULT_FETCH_OPTIONS = DomainFetchOptions()


def _http_failure_reason(exc: aiohttp.ClientResponseError) -> str:
    if exc.status == _NOT_FOUND_STATUS:
        return FAILURE_NOT_FOUND
    return f"http_error_{exc.status}"


def _fragments_from_locations(locations: Iterable[dict[str, Any]]) -> tuple[DomainFragment, ...]:
    fragments: list[DomainFragment] = []
    for location in locations:
        for fragment in location.get("fragments", []):
            start = fragment.get("start")
            end = fragment.get("end")
            if start is None or end is None:
                continue
            start_int, end_int = int(start), int(end)
            if end_int < start_int:
                continue
            fragments.append(DomainFragment(start=start_int, end=end_int))
    return tuple(sorted(set(fragments)))


def _protein_block(result: dict[str, Any], accession: str) -> dict[str, Any] | None:
    """Return the per-protein block of an entry result (InterPro lower-cases accessions)."""
    for protein in result.get("proteins", []):
        if str(protein.get("accession", "")).upper() == accession.upper():
            return protein
    proteins = result.get("proteins") or []
    return proteins[0] if proteins else None


def entries_from_api_results(results: Iterable[dict[str, Any]], accession: str) -> tuple[DomainEntry, ...]:
    """Convert ``entry/all/protein/uniprot`` results into domain entries."""
    entries: list[DomainEntry] = []
    for result in results:
        metadata = result.get("metadata") or {}
        entry_accession = str(metadata.get("accession", ""))
        if not entry_accession:
            continue
        protein = _protein_block(result, accession)
        if protein is None:
            continue
        fragments = _fragments_from_locations(protein.get("entry_protein_locations") or [])
        if not fragments:
            continue
        entries.append(
            DomainEntry(
                accession=entry_accession,
                source_database=str(metadata.get("source_database", "")),
                entry_type=str(metadata.get("type", "")),
                name=str(metadata.get("name", "")),
                integrated=str(metadata.get("integrated") or ""),
                fragments=fragments,
            ),
        )
    entries.sort(key=lambda entry: (entry.start, entry.end, entry.accession))
    return tuple(entries)


def record_from_api_payloads(
    accession: str,
    entry_results: Iterable[dict[str, Any]],
    protein_payload: dict[str, Any] | None,
) -> ProteinDomainRecord:
    """Assemble a protein domain record from the two InterPro API payloads."""
    entries = entries_from_api_results(entry_results, accession)
    metadata: dict[str, Any] = {}
    if protein_payload:
        metadata = protein_payload.get("metadata") or protein_payload

    organism = metadata.get("source_organism") or {}
    length = metadata.get("length") or 0
    return ProteinDomainRecord(
        accession=accession,
        name=str(metadata.get("name", "")),
        length=int(length),
        sequence=str(metadata.get("sequence") or ""),
        source_database=str(metadata.get("source_database", "")),
        organism_tax_id=str(organism.get("taxId", "")),
        organism_name=str(organism.get("fullName") or organism.get("scientificName") or ""),
        ida_accession=str(metadata.get("ida_accession") or ""),
        is_fragment=bool(metadata.get("is_fragment", False)),
        entries=entries,
    )


async def _get_all_pages(
    session: aiohttp.ClientSession,
    url: str,
    limiter: RateLimiter | None = None,
) -> list[dict[str, Any]]:
    """Follow InterPro cursor pagination and return every result object."""
    results: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        if limiter is not None:
            await limiter.acquire()
        payload = await get_with_retry(session, next_url)
        results.extend(payload.get("results", []))
        next_url = payload.get("next")
    return results


async def fetch_protein_record(
    session: aiohttp.ClientSession,
    accession: str,
    *,
    include_sequence: bool = True,
    limiter: RateLimiter | None = None,
) -> tuple[ProteinDomainRecord | None, str]:
    """Fetch domains (and optionally the sequence) for one accession.

    Returns:
        Tuple of the record (``None`` on failure) and a failure reason (``""`` on success).
    """
    try:
        entry_results = await _get_all_pages(session, ENTRIES_URL_TEMPLATE.format(accession=accession), limiter)
        protein_payload: dict[str, Any] | None = None
        if include_sequence:
            if limiter is not None:
                await limiter.acquire()
            protein_payload = await get_with_retry(session, PROTEIN_URL_TEMPLATE.format(accession=accession))
    except aiohttp.ClientResponseError as exc:
        return None, _http_failure_reason(exc)
    except RuntimeError:
        return None, FAILURE_RETRIES_EXHAUSTED
    except aiohttp.ClientError as exc:
        return None, f"client_error_{type(exc).__name__}"

    if not entry_results and protein_payload is None:
        return None, FAILURE_EMPTY_RESPONSE

    return record_from_api_payloads(accession, entry_results, protein_payload), ""


async def fetch_domain_store(
    session: aiohttp.ClientSession,
    accessions: Sequence[str],
    *,
    options: DomainFetchOptions = DEFAULT_FETCH_OPTIONS,
    store: DomainStore | None = None,
) -> DomainStore:
    """Fetch domain records for many accessions into a (possibly resumed) store."""
    result_store = store if store is not None else DomainStore()
    semaphore = asyncio.Semaphore(max(1, options.concurrency))
    limiter = RateLimiter(options.requests_per_second)

    async def worker(accession: str) -> tuple[str, ProteinDomainRecord | None, str]:
        async with semaphore:
            record, reason = await fetch_protein_record(
                session,
                accession,
                include_sequence=options.include_sequence,
                limiter=limiter,
            )
            return accession, record, reason

    tasks = [asyncio.ensure_future(worker(accession)) for accession in accessions]
    completed = 0
    with tqdm(
        total=len(tasks),
        desc="Fetching domains",
        unit=" proteins",
        disable=not options.show_progress,
    ) as progress:
        for future in asyncio.as_completed(tasks):
            accession, record, reason = await future
            if record is None:
                result_store.add_failure(accession, reason or FAILURE_EMPTY_RESPONSE)
            else:
                result_store.add(record)
            completed += 1
            progress.update(1)
            if (
                options.checkpoint_path is not None
                and options.checkpoint_every > 0
                and completed % options.checkpoint_every == 0
            ):
                write_domain_store(result_store, options.checkpoint_path)

    return result_store


def accessions_from_fetch_json(path: Path) -> list[str]:
    """Extract unique accessions from a DnaK or DnaJ InterPro fetch JSON file.

    Raises:
        ValueError: If the file is not valid JSON.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc.msg}"
        raise ValueError(msg) from exc

    for warning in fetch_warnings(data):
        logger.warning("%s", warning)
    return extract_accessions(data)


def accessions_from_file(path: Path) -> list[str]:
    """Read one accession per line, dropping blanks and duplicates (first wins)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        accession = normalize_accession(line.split("\t", maxsplit=1)[0])
        if accession and accession not in seen:
            seen.add(accession)
            ordered.append(accession)
    return ordered


def select_accession_slice(accessions: list[str], *, start: int, count: int | None) -> list[str]:
    """Return a 1-based contiguous slice for SLURM array chunking.

    Raises:
        ValueError: If ``start`` is below 1 or ``count`` is not positive.
    """
    if start < 1:
        msg = f"--start must be >= 1 (got {start})"
        raise ValueError(msg)
    if count is not None and count < 1:
        msg = f"--count must be >= 1 (got {count})"
        raise ValueError(msg)
    begin = start - 1
    end = None if count is None else begin + count
    return accessions[begin:end]


def pending_accessions(
    requested: Sequence[str],
    store: DomainStore,
    *,
    want_sequences: bool = True,
) -> list[str]:
    """Return the accessions still needing a fetch.

    An accession already in the store is skipped, *unless* this run wants sequences and
    the stored record has none: a store built with ``--no-sequence`` would otherwise be
    permanently sequence-less, because every later run would treat those accessions as
    already done.
    """
    pending: list[str] = []
    for accession in requested:
        record = store.proteins.get(accession)
        if record is None or (want_sequences and not record.has_sequence):
            pending.append(accession)
    return pending


async def backfill_metadata(
    session: aiohttp.ClientSession,
    store: DomainStore,
    *,
    options: DomainFetchOptions = DEFAULT_FETCH_OPTIONS,
) -> int:
    """Refresh protein metadata (notably the fragment flag) for an existing store.

    Domain matches are the expensive half of a fetch, so a store built before a metadata
    field existed should not be rebuilt from scratch just to gain it. This re-requests
    only the protein endpoint, one request per accession, and updates the records in
    place.

    Returns:
        How many records were updated.
    """
    limiter = RateLimiter(options.requests_per_second)
    semaphore = asyncio.Semaphore(max(1, options.concurrency))
    accessions = sorted(store.proteins)

    async def worker(accession: str) -> tuple[str, dict[str, Any] | None]:
        async with semaphore:
            await limiter.acquire()
            try:
                payload = await get_with_retry(session, PROTEIN_URL_TEMPLATE.format(accession=accession))
            except (aiohttp.ClientError, RuntimeError):
                return accession, None
            return accession, payload

    updated = 0
    tasks = [asyncio.ensure_future(worker(accession)) for accession in accessions]
    with tqdm(
        total=len(tasks), desc="Backfilling metadata", unit=" proteins", disable=not options.show_progress
    ) as bar:
        for future in asyncio.as_completed(tasks):
            accession, payload = await future
            bar.update(1)
            if payload is None:
                continue
            metadata = payload.get("metadata") or payload
            record = store.proteins[accession]
            store.proteins[accession] = replace(
                record,
                is_fragment=bool(metadata.get("is_fragment", record.is_fragment)),
            )
            updated += 1
    return updated


def merge_domain_stores(paths: Iterable[Path]) -> DomainStore:
    """Merge per-chunk domain stores into one store."""
    merged = DomainStore(source="merged")
    for path in sorted(paths):
        merged.merge(load_domain_store(path))
    return merged


def _resolve_accessions(args: argparse.Namespace) -> list[str]:
    if args.from_fetch_json is not None:
        accessions = accessions_from_fetch_json(args.from_fetch_json)
    elif args.accessions_file is not None:
        accessions = accessions_from_file(args.accessions_file)
    else:
        accessions = [normalized for value in args.accessions if (normalized := normalize_accession(value)) is not None]
    return select_accession_slice(accessions, start=args.start, count=args.count)


def _load_resume_store(args: argparse.Namespace) -> DomainStore:
    if args.refresh:
        return DomainStore()
    for candidate in (args.output, args.checkpoint):
        if candidate is not None and candidate.is_file():
            store = load_domain_store(candidate)
            logger.info("Resuming from %s with %s existing record(s).", candidate, len(store))
            return store
    return DomainStore()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fetch-protein-domains`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch full InterPro domain annotations and sequences for UniProt accessions.",
    )
    parser.add_argument("accessions", nargs="*", help="UniProt accessions (alternative to --from-fetch-json)")
    parser.add_argument(
        "--from-fetch-json",
        type=Path,
        default=None,
        help="DnaK or DnaJ InterPro fetch JSON to take accessions from",
    )
    parser.add_argument(
        "--accessions-file",
        type=Path,
        default=None,
        help="Text file with one accession per line (e.g. incomplete_accessions.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Domain store output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help=f"Max concurrent accession fetches (default: {_DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--start", type=int, default=1, help="1-based index of the first accession to fetch")
    parser.add_argument("--count", type=int, default=None, help="How many accessions to fetch from --start")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint store path for resume (default: <output>.checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=_DEFAULT_CHECKPOINT_EVERY,
        help=f"Write the checkpoint every N accessions (default: {_DEFAULT_CHECKPOINT_EVERY})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help=f"Total request timeout in seconds (default: {_DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=_DEFAULT_REQUESTS_PER_SECOND,
        help=(
            f"Max InterPro requests per second across this process (default: "
            f"{_DEFAULT_REQUESTS_PER_SECOND}); 0 disables pacing"
        ),
    )
    parser.add_argument(
        "--no-sequence",
        action="store_true",
        help="Skip the sequence/organism request (domains only)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing output/checkpoint and re-fetch every accession",
    )
    parser.add_argument(
        "--backfill-metadata",
        action="store_true",
        help="Refresh protein metadata (e.g. the fragment flag) for an existing --output store",
    )
    parser.add_argument(
        "--merge-stores",
        type=Path,
        default=None,
        help="Merge every *.json domain store in this directory into --output and exit",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable the progress bar")
    return parser


def _run_merge(args: argparse.Namespace) -> None:
    paths = sorted(path for path in args.merge_stores.glob("*.json") if path != args.output)
    merged = merge_domain_stores(paths)
    write_domain_store(merged, args.output)
    sys.stderr.write(
        f"Merged {len(paths)} store file(s) into {args.output}: "
        f"{len(merged)} protein(s), {len(merged.failures)} failure(s).\n",
    )


def _requested_accessions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    """Validate the input selectors and resolve them to an accession list."""
    if not any([args.accessions, args.from_fetch_json, args.accessions_file]):
        parser.error("Provide accessions positionally, or use --from-fetch-json / --accessions-file.")
    for path, label in ((args.from_fetch_json, "fetch JSON"), (args.accessions_file, "accessions file")):
        if path is not None and not path.is_file():
            parser.error(f"{label} not found: {path}")

    try:
        requested = _resolve_accessions(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not requested:
        parser.error("No accessions selected.")
    return requested


async def _run_backfill(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Refresh metadata for an existing store in place."""
    if not args.output.is_file():
        parser.error(f"Store to backfill not found: {args.output}")
    try:
        store = load_domain_store(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
        updated = await backfill_metadata(
            session,
            store,
            options=DomainFetchOptions(
                concurrency=args.concurrency,
                show_progress=not args.quiet,
                requests_per_second=args.rate_limit,
            ),
        )
    write_domain_store(store, args.output)
    fragments = sum(1 for record in store if record.is_fragment)
    sys.stderr.write(
        f"Backfilled {updated} of {len(store)} record(s) in {args.output}; {fragments} flagged as fragments.\n",
    )


async def main(argv: list[str] | None = None) -> None:
    """Fetch domain annotations for every requested accession."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.backfill_metadata:
        await _run_backfill(args, parser)
        return

    if args.merge_stores is not None:
        if not args.merge_stores.is_dir():
            parser.error(f"--merge-stores directory not found: {args.merge_stores}")
        _run_merge(args)
        return

    requested = _requested_accessions(args, parser)

    checkpoint = args.checkpoint if args.checkpoint is not None else args.output.with_suffix(".checkpoint.json")
    try:
        store = _load_resume_store(args)
    except ValueError as exc:
        parser.error(str(exc))
    store.source = str(args.from_fetch_json or args.accessions_file or "cli")

    pending = pending_accessions(requested, store, want_sequences=not args.no_sequence)
    logger.info(
        "Fetching domains for %s accession(s) (%s already present, %s requested).",
        len(pending),
        len(requested) - len(pending),
        len(requested),
    )

    if pending:
        options = DomainFetchOptions(
            concurrency=args.concurrency,
            include_sequence=not args.no_sequence,
            checkpoint_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            show_progress=not args.quiet,
            requests_per_second=args.rate_limit,
        )
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
            store = await fetch_domain_store(session, pending, options=options, store=store)

    write_domain_store(store, args.output)
    checkpoint.unlink(missing_ok=True)

    with_sequence = sum(1 for record in store if record.has_sequence)
    sys.stderr.write(
        f"Wrote {args.output}: {len(store)} protein record(s), "
        f"{with_sequence} with sequence, {len(store.failures)} failure(s).\n",
    )


def cli() -> None:
    """Console script entry point for fetch-protein-domains."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
