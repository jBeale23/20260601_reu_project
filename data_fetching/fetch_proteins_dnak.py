#!/usr/bin/env python3


"""Fetches all proteins for InterPro entry IPR012725 (DnaK).

This entry has no domain architecture groups. The script retrieves all proteins
that match the entry directly using the InterPro protein API.
No deduplication is performed: proteins are stored exactly as returned by the API.

Output format::

    {
        "entry_accession": "IPR012725",
        "proteins_reported": 100000,
        "total_proteins_fetched": 100000,
        "proteins": [...],
        "note": "No deduplication - proteins are stored exactly as returned by the API.",
    }
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Type aliases ---
# Raw JSON object returned by the InterPro API — keys are strings, values are
# JSON-native scalars or nested containers.
type ApiResponse = dict[str, str | int | float | bool | list | dict | None]

# Assembled output dict produced by this script, written to the output file.
type ProteinResult = dict[str, str | int | list[ApiResponse]]

# --- Constants (used as defaults in main) ---
ENTRY_ACCESSION = "IPR012725"

_DEFAULT_PAGE_SIZE = 200
_DEFAULT_OUTPUT = Path("ipr012725_proteins.json")
_DEFAULT_MAX_PERCENT = 0.01
_DEFAULT_MAX_ABS = 100

HEADERS = {"User-Agent": "research-script/1.0 (tmarena1@jh.edu)"}

# HTTP status codes that warrant a retry
RATE_LIMIT_STATUSES = {408, 429, 503}
EXPIRED_CURSOR_STATUS = 404


def make_protein_url(entry_accession: str, page_size: int) -> str:
    """Build the InterPro API URL for fetching proteins by entry accession.

    Args:
        entry_accession: The InterPro entry accession (in this case, IPR012725).
        page_size: Number of results to request per page.

    Returns:
        Fully formatted API URL string.
    """
    return (
        f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/"
        f"entry/InterPro/{entry_accession}/?page_size={page_size}&format=json"
    )


async def get_with_retry(session: aiohttp.ClientSession, url: str) -> ApiResponse:
    """Fetch a URL with exponential backoff on rate limits and expired cursors.

    Args:
        session: The aiohttp client session to use.
        url: The URL to fetch.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If all 5 retry attempts are exhausted.
    """
    for attempt in range(5):
        async with session.get(url) as resp:
            if resp.status in RATE_LIMIT_STATUSES:
                wait = min(60 * (2**attempt), 300)
                logger.warning("Rate limited (%s), waiting %ss before retrying...", resp.status, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status == EXPIRED_CURSOR_STATUS:
                # Cursors can expire mid-pagination on the InterPro API.
                # A short wait usually resolves it.
                wait = min(10 * (2**attempt), 60)
                logger.warning("Cursor expired (404), attempt %s/5, retrying in %ss...", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return await resp.json()
    msg = f"Still failing after 5 retries: {url}"
    raise RuntimeError(msg)


def validate_api_response(data: ApiResponse) -> None:
    """Raise if the API response is missing expected top-level keys.

    Acts as a lightweight runtime guard against unexpected API format changes.

    Args:
        data: Parsed JSON response from the InterPro API.

    Raises:
        RuntimeError: If any required keys are absent from the response.
    """
    required = {"results", "next", "count"}
    missing = required - data.keys()
    if missing:
        msg = f"Unexpected API response format — missing keys: {missing}"
        raise RuntimeError(msg)


def check_count_anomaly(
    reported: int,
    fetched: int,
    max_abs: int = _DEFAULT_MAX_ABS,
    max_percent: float = _DEFAULT_MAX_PERCENT,
) -> None:
    """Log a warning if the fetched protein count diverges too far from reported.

    Both thresholds must be exceeded simultaneously to trigger the warning,
    to avoid false positives on very large or very small datasets.

    Args:
        reported: Protein count as reported by the API.
        fetched: Protein count actually retrieved.
        max_abs: Maximum allowed absolute difference before warning.
        max_percent: Maximum allowed fractional difference before warning (e.g. 0.01 = 1%).
    """
    if reported <= 0:
        return
    abs_diff = abs(reported - fetched)
    pct_diff = abs_diff / reported
    if abs_diff > max_abs and pct_diff > max_percent:
        logger.warning(
            "Count anomaly: reported=%s fetched=%s (Δ%s, %.1f%%)",
            reported,
            fetched,
            abs_diff,
            pct_diff * 100,
        )


async def fetch_all_proteins(
    session: aiohttp.ClientSession,
    url: str,
    first_page: ApiResponse | None = None,
) -> ProteinResult:
    """Fetch all proteins for the entry via cursor-based pagination.

    Pages must be fetched sequentially because each page's URL is returned
    by the previous response. If a pre-fetched first page is provided, it is
    used directly without making an additional request, avoiding a redundant
    API call. If retries are exhausted mid-pagination, the partial results
    collected so far are returned rather than crashing the entire run.

    Args:
        session: The aiohttp client session to use.
        url: The initial API URL to fetch from.
        first_page: Optional pre-fetched first page to reuse.

    Returns:
        Dictionary containing entry accession, reported count, fetched count,
        and the full list of protein results.
    """
    proteins: list[ApiResponse] = []
    total_count: int | None = None

    with tqdm(desc="Fetching proteins", unit=" proteins", leave=False) as pbar:
        while url:
            try:
                data = first_page if first_page is not None else await get_with_retry(session, url)
            except RuntimeError:
                # If retries are exhausted mid-pagination, return partial results
                # rather than crashing and losing everything collected so far.
                logger.exception(
                    "Retries exhausted mid-pagination, returning %s proteins collected so far",
                    len(proteins),
                )
                break
            first_page = None  # only use the pre-fetched page once
            if total_count is None:
                total_count = data.get("count")
                if total_count is not None:
                    pbar.total = total_count
                    pbar.refresh()
            batch: list[ApiResponse] = data.get("results", [])
            proteins.extend(batch)
            pbar.update(len(batch))
            url = data.get("next")

    return {
        "entry_accession": ENTRY_ACCESSION,
        "proteins_reported": total_count,
        "proteins_fetched": len(proteins),
        "proteins": proteins,
    }


def _write_output(data: ProteinResult, output_file: Path) -> None:
    """Write output data to disk synchronously (internal helper for asyncio.to_thread).

    Args:
        data: The full output dictionary to serialize.
        output_file: Path to write the JSON output to.
    """
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)


async def main() -> None:
    """Fetch all proteins for InterPro entry IPR012725 (DnaK).

    Proteins are fetched sequentially via cursor-based pagination.
    A count anomaly check is performed after fetching to warn if the
    number of proteins retrieved diverges significantly from what the
    API reported. Results are written to a JSON file.
    """
    parser = argparse.ArgumentParser(description="Fetch proteins for IPR012725 (DnaK).")
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        default=None,
        help=f"Page size to request from the API (default: {_DEFAULT_PAGE_SIZE})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON file path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-m",
        "--max-percent",
        type=float,
        default=None,
        help=f"Maximum fractional difference allowed before warning (default: {_DEFAULT_MAX_PERCENT})",
    )
    parser.add_argument(
        "-a",
        "--max-abs",
        type=int,
        default=None,
        help=f"Maximum absolute difference allowed before warning (default: {_DEFAULT_MAX_ABS})",
    )
    args = parser.parse_args()

    # Apply defaults idiomatically
    page_size = args.page_size if args.page_size is not None else _DEFAULT_PAGE_SIZE
    output_file = args.output if args.output is not None else _DEFAULT_OUTPUT
    max_percent = args.max_percent if args.max_percent is not None else _DEFAULT_MAX_PERCENT
    max_abs = args.max_abs if args.max_abs is not None else _DEFAULT_MAX_ABS

    protein_url = make_protein_url(ENTRY_ACCESSION, page_size)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        logger.info("Fetching proteins for InterPro entry %s...", ENTRY_ACCESSION)
        first_page = await get_with_retry(session, protein_url)
        validate_api_response(first_page)  # guard against unexpected API format changes
        protein_data = await fetch_all_proteins(session, protein_url, first_page=first_page)

    reported = protein_data.get("proteins_reported") or 0
    fetched = protein_data.get("proteins_fetched", 0)
    logger.info("Proteins reported: %s | Proteins fetched: %s", reported, fetched)
    check_count_anomaly(reported, fetched, max_abs=max_abs, max_percent=max_percent)

    output_data: ProteinResult = {
        "entry_accession": ENTRY_ACCESSION,
        "proteins_reported": reported,
        "total_proteins_fetched": fetched,
        "proteins": protein_data.get("proteins", []),
        "note": (
            "No deduplication - proteins are stored exactly as returned by the API. "
            "Duplicate accession values are possible only if the API returns duplicates."
        ),
    }

    # offload blocking I/O off the event loop
    await asyncio.to_thread(_write_output, output_data, output_file)
    logger.info("Results written to %s", output_file)


if __name__ == "__main__":
    asyncio.run(main())
