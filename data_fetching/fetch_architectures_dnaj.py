"""Fetches all proteins for the first N domain architectures for InterPro entry IPR001623 (DnaJ/HSP40).

NO deduplication, duplicate flagging:
- Stores proteins exactly as they are fetched for each architecture
- If a protein appears in multiple architectures, it will be stored multiple times
- Each protein instance is flagged with how many architectures it appears in total
- Includes comprehensive sanity checks

Output format::

    {
        "sanity_checks": {
            "total_architectures_fetched": 20,
            "total_architectures_expected": 20,
            "architectures_match": true,
            "total_proteins_fetched": 225610,
            "total_unique_proteins": 156000,
            "duplicate_proteins": 69610,
            "average_architectures_per_protein": 1.44,
            "all_architectures_present": true,
            "no_zero_count_architectures": true,
        },
        "architectures": [
            {
                "ida": "PF00226:IPR001623",
                "ida_id": "500088c3adc88e8af670fe08554083396acf46f3",
                "unique_proteins_reported": 98256,
                "proteins_fetched": 99077,
                "proteins": [{"appears_in_architecture_count": 2}],
            }
        ],
    }
"""

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

ARCH_URL = "https://www.ebi.ac.uk/interpro/api/entry/InterPro/IPR001623/?ida&page_size=20&format=json"

PROTEIN_URL_TEMPLATE = (
    "https://www.ebi.ac.uk/interpro/api/protein/UniProt/"
    "entry/InterPro/IPR001623/?ida={ida_id}&page_size=200&format=json"
)

HEADERS = {"User-Agent": "research-script/1.0 (tmarena1@jh.edu)"}

N_ARCHITECTURES = 20
CONCURRENCY_LIMIT = 5
OUTPUT_FILE = Path("ipr001623_domain_architectures_no_dedup.json")

# Known values from the InterPro website, used to confirm we start at architecture #1
EXPECTED_FIRST_IDA_ID = "500088c3adc88e8af670fe08554083396acf46f3"
EXPECTED_FIRST_PROTEIN_COUNT = 98256

# HTTP status codes that warrant a retry
RATE_LIMIT_STATUSES = {408, 429, 503}
EXPIRED_CURSOR_STATUS = 404


async def get_with_retry(session: aiohttp.ClientSession, url: str) -> dict:
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
                wait = 61 * (attempt + 1)
                logger.warning("Rate limited (%s), waiting %ss before retrying...", resp.status, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status == EXPIRED_CURSOR_STATUS:
                # Cursors can expire mid-pagination on the InterPro API.
                # A short wait usually resolves it.
                wait = 10 * (attempt + 1)
                logger.warning("Cursor expired (404), attempt %s/5, retrying in %ss...", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return await resp.json()
    msg = f"Still failing after 5 retries: {url}"
    raise RuntimeError(msg)


def verify_first_architecture(architectures: list) -> None:
    """Confirm the API is returning architectures starting from #1, not #2 or later.

    Compares the first result against known values scraped from the InterPro
    website. Raises an error and halts the script if the IDA ID doesn't match.

    Args:
        architectures: List of architecture dicts returned by the API.

    Raises:
        RuntimeError: If the API returned zero results or the wrong first architecture.
    """
    if not architectures:
        msg = "API returned zero architectures — something is wrong."
        raise RuntimeError(msg)

    first = architectures[0]
    actual_ida_id = first.get("ida_id")
    actual_count = first.get("unique_proteins", 0)

    logger.info("Index check — first architecture returned by API:")
    logger.info("  IDA ID:         %s", actual_ida_id)
    logger.info("  Protein count:  %s", actual_count)
    logger.info("  Expected IDA:   %s", EXPECTED_FIRST_IDA_ID)
    logger.info("  Expected count: ~%s", EXPECTED_FIRST_PROTEIN_COUNT)

    if actual_ida_id != EXPECTED_FIRST_IDA_ID:
        msg = (
            f"First architecture mismatch — we may be starting at the wrong index.\n"
            f"  Got:      {actual_ida_id}\n"
            f"  Expected: {EXPECTED_FIRST_IDA_ID}\n"
            f"Double-check the API pagination offset."
        )
        raise RuntimeError(msg)

    # Allow a 5% margin since the database updates periodically
    margin = EXPECTED_FIRST_PROTEIN_COUNT * 0.05
    if abs(actual_count - EXPECTED_FIRST_PROTEIN_COUNT) > margin:
        logger.warning(
            "Protein count differs from expected by more than 5%% (%s vs %s). The database may have been updated.",
            actual_count,
            EXPECTED_FIRST_PROTEIN_COUNT,
        )
    else:
        logger.info("✓ Index check passed — starting from architecture #1 as expected.")


async def fetch_proteins_for_arch(
    session: aiohttp.ClientSession,
    arch: dict,
    index: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Fetch all proteins for a single domain architecture via cursor pagination.

    Pages must be fetched sequentially because each page's URL is returned
    by the previous response. No deduplication is performed.

    Args:
        session: The aiohttp client session to use.
        arch: Architecture metadata dict from the InterPro API.
        index: 1-based index of this architecture in the run (for display).
        semaphore: Shared semaphore limiting concurrent architecture fetches.

    Returns:
        Dictionary containing architecture metadata and all fetched proteins.
    """
    ida_id = arch.get("ida_id")
    proteins = []
    url = PROTEIN_URL_TEMPLATE.format(ida_id=ida_id)

    async with semaphore:
        with tqdm(
            desc=f"Arch {index}/{N_ARCHITECTURES} ({ida_id[:8]}...)",
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


def _write_output(data: dict) -> None:
    """Write output data to disk synchronously.

    Args:
        data: The full output dictionary to serialize.
    """
    with OUTPUT_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def run_sanity_checks(
    architectures: list,
    expected_count: int,
    protein_counts: dict,
) -> dict:
    """Run comprehensive sanity checks on the fetched data.

    Args:
        architectures: List of architecture result dicts.
        expected_count: How many architectures we intended to fetch.
        protein_counts: Dict mapping protein accession to occurrence count.

    Returns:
        Dictionary of sanity check results and statistics.
    """
    fetched_arch_count = len(architectures)
    total_fetched = sum(r["proteins_fetched"] for r in architectures)
    total_reported = sum(r["unique_proteins_reported"] for r in architectures)
    unique_proteins = len(protein_counts)
    duplicate_entries = total_fetched - unique_proteins

    all_present = fetched_arch_count == expected_count
    zero_count_archs = [a for a in architectures if a["proteins_fetched"] == 0]
    no_zero_counts = len(zero_count_archs) == 0
    avg_archs_per_protein = total_fetched / unique_proteins if unique_proteins > 0 else 0
    multi_arch_proteins = sum(1 for count in protein_counts.values() if count > 1)

    return {
        "total_architectures_fetched": fetched_arch_count,
        "total_architectures_expected": expected_count,
        "architectures_match": all_present,
        "total_proteins_fetched": total_fetched,
        "total_proteins_reported_by_api": total_reported,
        "total_unique_proteins": unique_proteins,
        "duplicate_protein_entries": duplicate_entries,
        "proteins_in_multiple_architectures": multi_arch_proteins,
        "average_architectures_per_protein": round(avg_archs_per_protein, 2),
        "all_architectures_present": all_present,
        "no_zero_count_architectures": no_zero_counts,
        "sanity_check_passed": all_present and no_zero_counts,
    }


def print_sanity_report(checks: dict) -> None:
    """Log a formatted summary of the sanity check results.

    Args:
        checks: Dictionary returned by run_sanity_checks.
    """
    logger.info("=" * 70)
    logger.info("SANITY CHECK REPORT")
    logger.info("=" * 70)

    if checks["sanity_check_passed"]:
        logger.info("✓ All sanity checks PASSED")
    else:
        logger.error("✗ SANITY CHECK FAILED - Review issues below")

    logger.info("Architectures:")
    logger.info("  Expected:  %s", checks["total_architectures_expected"])
    logger.info("  Fetched:   %s", checks["total_architectures_fetched"])
    logger.info("  Status:    %s", "✓ OK" if checks["architectures_match"] else "✗ MISMATCH")

    logger.info("Proteins:")
    logger.info("  Total fetched:        %8s", checks["total_proteins_fetched"])
    logger.info("  API reported:         %8s", checks["total_proteins_reported_by_api"])
    logger.info("  Unique proteins:      %8s", checks["total_unique_proteins"])
    logger.info("  Duplicate entries:    %8s", checks["duplicate_protein_entries"])
    logger.info("  Multi-arch proteins:  %8s", checks["proteins_in_multiple_architectures"])

    logger.info("Duplicate Analysis:")
    logger.info("  Avg architectures per protein:  %s", checks["average_architectures_per_protein"])
    logger.info("  Proteins in 2+ architectures:   %6s", checks["proteins_in_multiple_architectures"])

    logger.info("Validation:")
    logger.info("  All architectures present:     %s", "✓ Yes" if checks["all_architectures_present"] else "✗ No")
    logger.info("  No zero-count architectures:   %s", "✓ Yes" if checks["no_zero_count_architectures"] else "✗ No")
    logger.info("=" * 70)


async def main() -> None:
    """Fetch all proteins for the top N domain architectures of IPR001623.

    Architectures are fetched concurrently up to CONCURRENCY_LIMIT.
    Within each architecture, pages are fetched sequentially due to
    cursor-based pagination. Results are written to a JSON file.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        logger.info("Fetching architecture groups from InterPro...")
        arch_data = await get_with_retry(session, ARCH_URL)
        architectures = arch_data.get("results", [])[:N_ARCHITECTURES]
        logger.info("Got %s architecture group(s).", len(architectures))

        # Confirm we're starting from architecture #1 before doing anything else.
        # If this raises, the script stops here and nothing gets written to disk.
        verify_first_architecture(architectures)

        logger.info("Fetching proteins...")

        # Each architecture is independent so we fire them all at once.
        # The semaphore keeps us from overwhelming the EBI API.
        tasks = [fetch_proteins_for_arch(session, arch, i, semaphore) for i, arch in enumerate(architectures, start=1)]
        all_results = await asyncio.gather(*tasks)

    # Figure out which proteins appear in more than one architecture
    logger.info("Processing duplicate flags...")
    protein_occurrence_count: dict = {}

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

    logger.info("Running sanity checks...")
    sanity_checks = run_sanity_checks(all_results, N_ARCHITECTURES, protein_occurrence_count)

    total_fetched = sum(r["proteins_fetched"] for r in all_results)

    output_data = {
        "sanity_checks": sanity_checks,
        "architectures": list(all_results),
        "total_proteins_fetched": total_fetched,
        "note": (
            "NO deduplication - proteins may appear multiple times if present in "
            "multiple architectures. Use 'appears_in_architecture_count' to identify duplicates."
        ),
    }

    # Write to disk using a helper so the blocking I/O stays out of the async context
    await asyncio.to_thread(_write_output, output_data)

    print_sanity_report(sanity_checks)
    logger.info("Results written to %s", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
