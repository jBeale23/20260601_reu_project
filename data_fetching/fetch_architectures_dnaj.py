"""Fetches all proteins for the first N domain architectures for InterPro entry IPR001623 (DnaJ/HSP40).

No deduplication, duplicate flagging:
- Stores proteins exactly as they are fetched for each architecture
- If a protein appears in multiple architectures, it will be stored multiple times
- Each protein instance is flagged with how many architectures it appears in total

Output format::

    {
        "architectures": [
            {
                "ida": "PF00226:IPR001623",
                "ida_id": "500088c3adc88e8af670fe08554083396acf46f3",
                "unique_proteins_reported": 98256,
                "proteins_fetched": 99077,
                "proteins": [{"appears_in_architecture_count": 2}],
            }
        ],
        "total_proteins_fetched": 225610,
        "note": "No deduplication - proteins may appear multiple times...",
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
type ArchResult = dict[str, object]
type ProteinCounts = dict[str, int]

# --- Constants (used as defaults in main) ---
_DEFAULT_ARCH_URL = "https://www.ebi.ac.uk/interpro/api/entry/InterPro/IPR001623/?ida&page_size=20&format=json"
_DEFAULT_N_ARCHITECTURES = 20
_DEFAULT_CONCURRENCY = 5
_DEFAULT_OUTPUT = Path("ipr001623_domain_architectures_no_dedup.json")

PROTEIN_URL_TEMPLATE = (
    "https://www.ebi.ac.uk/interpro/api/protein/UniProt/"
    "entry/InterPro/IPR001623/?ida={ida_id}&page_size=200&format=json"
)
HEADERS = {"User-Agent": "research-script/1.0 (tmarena1@jh.edu)"}

# Known values from the InterPro website, used to confirm we start at architecture #1
EXPECTED_FIRST_IDA_ID = "500088c3adc88e8af670fe08554083396acf46f3"
EXPECTED_FIRST_PROTEIN_COUNT = 98256

# HTTP status codes that warrant a retry
RATE_LIMIT_STATUSES = {408, 429, 503}
EXPIRED_CURSOR_STATUS = 404


async def get_with_retry(session: aiohttp.ClientSession, url: str) -> ArchResult:
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


def validate_api_response(data: ArchResult) -> None:
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


async def fetch_proteins_for_arch(
    session: aiohttp.ClientSession,
    arch: ArchResult,
    index: int,
    semaphore: asyncio.Semaphore,
    n_architectures: int,
) -> ArchResult:
    """Fetch all proteins for a single domain architecture via cursor pagination.

    Pages must be fetched sequentially because each page's URL is returned
    by the previous response. No deduplication is performed.

    Args:
        session: The aiohttp client session to use.
        arch: Architecture metadata dict from the InterPro API.
        index: 1-based index of this architecture in the run (for display).
        semaphore: Shared semaphore limiting concurrent architecture fetches.
        n_architectures: Total number of architectures being fetched (for display).

    Returns:
        Dictionary containing architecture metadata and all fetched proteins.
    """
    ida_id = arch.get("ida_id")
    proteins: list[ArchResult] = []
    url = PROTEIN_URL_TEMPLATE.format(ida_id=ida_id)

    async with semaphore:
        with tqdm(
            desc=f"Arch {index}/{n_architectures} ({ida_id[:8]}...)",
            unit=" proteins",
            leave=False,
        ) as pbar:
            while url:
                try:
                    data = await get_with_retry(session, url)
                except RuntimeError:
                    # If retries are exhausted, skip this architecture rather
                    # than crashing the entire run and losing all other results.
                    logger.exception("Skipping arch %s (%s) after repeated failures", index, ida_id)
                    break
                batch = data.get("results", [])
                proteins.extend(batch)
                pbar.update(len(batch))
                url = data.get("next")

    return {
        "ida": arch.get("ida"),
        "ida_id": ida_id,
        "unique_proteins_reported": arch.get("unique_proteins", 0),
        "proteins_fetched": len(proteins),
        "proteins": proteins,
    }


def _write_output(data: ArchResult, output_file: Path) -> None:
    """Write output data to disk synchronously (internal helper for asyncio.to_thread).

    Args:
        data: The full output dictionary to serialize.
        output_file: Path to write the JSON output to.
    """
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)


async def main() -> None:
    """Fetch all proteins for the top N domain architectures of IPR001623.

    Architectures are fetched concurrently up to the concurrency limit.
    Within each architecture, pages are fetched sequentially due to
    cursor-based pagination. Results are written to a JSON file.
    """
    parser = argparse.ArgumentParser(description="Fetch IPR001623 domain architectures from InterPro.")
    parser.add_argument(
        "-n",
        "--n-architectures",
        type=int,
        default=None,
        help=f"Number of architectures to fetch (default: {_DEFAULT_N_ARCHITECTURES})",
    )
    parser.add_argument(
        "-u",
        "--arch-url",
        type=str,
        default=None,
        help="Architecture list URL",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=None,
        help=f"Max concurrent requests (default: {_DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON file path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # Apply defaults idiomatically
    n_architectures = args.n_architectures if args.n_architectures is not None else _DEFAULT_N_ARCHITECTURES
    arch_url = args.arch_url if args.arch_url is not None else _DEFAULT_ARCH_URL
    concurrency = args.concurrency if args.concurrency is not None else _DEFAULT_CONCURRENCY
    output_file = args.output if args.output is not None else _DEFAULT_OUTPUT

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        logger.info("Fetching architecture groups from InterPro...")
        arch_data = await get_with_retry(session, arch_url)
        validate_api_response(arch_data)  # guard against unexpected API format changes
        architectures = arch_data.get("results", [])[:n_architectures]
        logger.info("Got %s architecture group(s).", len(architectures))

        logger.info("Fetching proteins...")

        # Each architecture is independent so we send them off all at once.
        # The semaphore keeps us from overwhelming the EBI API.
        tasks = [
            fetch_proteins_for_arch(session, arch, i, semaphore, n_architectures)
            for i, arch in enumerate(architectures, start=1)
        ]
        all_results = await asyncio.gather(*tasks)

    # Figure out which proteins appear in more than one architecture
    logger.info("Processing duplicate flags...")
    protein_occurrence_count: ProteinCounts = {}

    for result in all_results:
        for protein in result["proteins"]:
            accession = protein.get("metadata", {}).get("accession")
            if accession:
                protein_occurrence_count[accession] = protein_occurrence_count.get(accession, 0) + 1

    # Tag each protein with how many architectures it shows up in
    for result in all_results:
        for protein in result["proteins"]:
            accession = protein.get("metadata", {}).get("accession")
            if accession:
                protein["appears_in_architecture_count"] = protein_occurrence_count[accession]

    total_fetched = sum(r["proteins_fetched"] for r in all_results)

    output_data: ArchResult = {
        "architectures": list(all_results),
        "total_proteins_fetched": total_fetched,
        "note": (
            "No deduplication - proteins may appear multiple times if present in "
            "multiple architectures. Use 'appears_in_architecture_count' to identify duplicates."
        ),
    }

    # offload blocking I/O off the event loop
    await asyncio.to_thread(_write_output, output_data, output_file)
    logger.info("Results written to %s", output_file)


if __name__ == "__main__":
    asyncio.run(main())
