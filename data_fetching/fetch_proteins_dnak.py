"""Fetches all proteins for InterPro entry IPR012725 (DnaK).

This entry has no domain architecture groups. The script retrieves all proteins
that match the entry directly using the InterPro protein API.
No deduplication is performed: proteins are stored exactly as returned by the API.

If the cursor chain times out mid-fetch, progress is saved to a checkpoint
file and the script exits cleanly. Re-running the script will resume from
the last successful cursor rather than starting over from the beginning.
Once the fetch completes successfully the checkpoint file is deleted.

Output format::

    {
        "entry_accession": "IPR012725",
        "proteins_reported": 100000,
        "total_proteins_fetched": 100000,
        "proteins": [...],
        "note": "...",
    }
"""

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
from tqdm import tqdm

from data_fetching.fetch_types import ApiResponse, ProteinResult
from data_fetching.utils import (
    INTERPRO_HEADERS,
    check_count_anomaly,
    checkpoint_exists,
    configure_logging,
    delete_checkpoint,
    get_with_retry,
    load_checkpoint,
    save_checkpoint,
    validate_api_response,
    write_output,
)

configure_logging()
logger = logging.getLogger(__name__)

ENTRY_ACCESSION = "IPR012725"

_DEFAULT_PAGE_SIZE = 200
_DEFAULT_OUTPUT = Path("ipr012725_proteins.json")
_DEFAULT_CHECKPOINT = Path("ipr012725_checkpoint.json")
_DEFAULT_MAX_PERCENT = 0.01
_DEFAULT_MAX_ABS = 100
_REQUEST_TIMEOUT = 300


def make_protein_url(entry_accession: str, page_size: int) -> str:
    """Build the first-page InterPro API URL for fetching proteins by entry accession.

    Args:
        entry_accession: The InterPro entry accession (e.g. IPR012725).
        page_size: Number of results to request per page.

    Returns:
        Fully formatted API URL string.
    """
    return (
        f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/"
        f"entry/InterPro/{entry_accession}/?page_size={page_size}&format=json"
    )


async def fetch_all_proteins(
    session: aiohttp.ClientSession,
    url: str,
    first_page: ApiResponse | None = None,
    checkpoint_file: Path | None = None,
) -> ProteinResult:
    """Fetch all proteins via sequential cursor pagination with checkpointing.

    If a checkpoint file exists from a previous interrupted run, pagination
    resumes from the last saved cursor rather than starting over. Progress
    is saved to the checkpoint file whenever pagination is interrupted so
    that re-running the script picks up exactly where it left off. The
    checkpoint file is deleted once the fetch completes successfully.

    Args:
        session: The aiohttp client session to use.
        url: The first-page URL to start from (used only on a fresh run).
        first_page: Optional pre-fetched first page to reuse (fresh run only).
        checkpoint_file: Path to read/write the checkpoint JSON, or None to
            disable checkpointing.

    Returns:
        Dictionary with entry accession, reported count, fetched count,
        all protein results, and ``is_partial`` flag.
    """
    proteins: list[ApiResponse] = []
    total_count: int | None = None
    is_partial = False
    next_url: str | None = None

    if checkpoint_file and await asyncio.to_thread(checkpoint_exists, checkpoint_file):
        checkpoint = await asyncio.to_thread(load_checkpoint, checkpoint_file)
        proteins = checkpoint.get("proteins", [])
        total_count = checkpoint.get("proteins_reported")
        next_url = checkpoint.get("next_url")
        logger.info(
            "Resuming from checkpoint: %s proteins collected so far, resuming at cursor %s",
            len(proteins),
            next_url,
        )

    if next_url is None:
        if first_page is None:
            try:
                first_page = await get_with_retry(session, url)
            except RuntimeError:
                logger.exception("Retries exhausted while fetching the first protein page.")
                return {
                    "entry_accession": ENTRY_ACCESSION,
                    "proteins_reported": None,
                    "proteins_fetched": 0,
                    "is_partial": True,
                    "proteins": [],
                }
        validate_api_response(first_page)
        proteins.extend(first_page.get("results", []))
        total_count = first_page.get("count")
        next_url = first_page.get("next")

    with tqdm(
        desc="Fetching proteins",
        total=total_count,
        initial=len(proteins),
        unit=" proteins",
        leave=False,
    ) as pbar:
        while next_url:
            try:
                data = await get_with_retry(session, next_url)
            except RuntimeError:
                logger.warning(
                    "Fetch interrupted at %s proteins — saving checkpoint to %s. "
                    "Re-run the script to resume from this point.",
                    len(proteins),
                    checkpoint_file,
                )
                if checkpoint_file:
                    await asyncio.to_thread(
                        save_checkpoint,
                        checkpoint_file,
                        {
                            "proteins_reported": total_count,
                            "next_url": next_url,
                            "proteins": proteins,
                        },
                    )
                    logger.info("Checkpoint saved to %s.", checkpoint_file)
                is_partial = True
                break
            validate_api_response(data)
            batch: list[ApiResponse] = data.get("results", [])
            proteins.extend(batch)
            pbar.update(len(batch))
            next_url = data.get("next")

    if not is_partial and checkpoint_file and await asyncio.to_thread(checkpoint_exists, checkpoint_file):
        await asyncio.to_thread(delete_checkpoint, checkpoint_file)
        logger.info("Fetch complete — checkpoint file deleted.")

    return {
        "entry_accession": ENTRY_ACCESSION,
        "proteins_reported": total_count,
        "proteins_fetched": len(proteins),
        "is_partial": is_partial,
        "proteins": proteins,
    }


async def main() -> None:
    """Fetch all proteins for InterPro entry IPR012725 (DnaK)."""
    parser = argparse.ArgumentParser(description="Fetch proteins for IPR012725 (DnaK).")
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        default=None,
        help=f"Page size to request from the API (default: {_DEFAULT_PAGE_SIZE})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help=f"Output JSON file path (default: {_DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "-k", "--checkpoint", type=Path, default=None, help=f"Checkpoint file path (default: {_DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "-m",
        "--max-percent",
        type=float,
        default=None,
        help=f"Percent difference threshold for soft notification (default: {_DEFAULT_MAX_PERCENT})",
    )
    parser.add_argument(
        "-a",
        "--max-abs",
        type=int,
        default=None,
        help=f"Absolute difference threshold for hard warning (default: {_DEFAULT_MAX_ABS})",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=None, help=f"Request timeout in seconds (default: {_REQUEST_TIMEOUT})"
    )
    args = parser.parse_args()

    page_size = args.page_size if args.page_size is not None else _DEFAULT_PAGE_SIZE
    output_file = args.output if args.output is not None else _DEFAULT_OUTPUT
    checkpoint_file = args.checkpoint if args.checkpoint is not None else _DEFAULT_CHECKPOINT
    max_percent = args.max_percent if args.max_percent is not None else _DEFAULT_MAX_PERCENT
    max_abs = args.max_abs if args.max_abs is not None else _DEFAULT_MAX_ABS
    timeout = args.timeout if args.timeout is not None else _REQUEST_TIMEOUT

    protein_url = make_protein_url(ENTRY_ACCESSION, page_size)
    session_timeout = aiohttp.ClientTimeout(total=timeout, sock_read=timeout)

    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=session_timeout) as session:
        logger.info("Fetching proteins for InterPro entry %s...", ENTRY_ACCESSION)

        first_page: ApiResponse | None = None
        if not await asyncio.to_thread(checkpoint_exists, checkpoint_file):
            first_page = await get_with_retry(session, protein_url)
            validate_api_response(first_page)

        protein_data = await fetch_all_proteins(
            session,
            protein_url,
            first_page=first_page,
            checkpoint_file=checkpoint_file,
        )

    reported = protein_data.get("proteins_reported") or 0
    fetched = protein_data.get("proteins_fetched", 0)
    logger.info("Proteins reported: %s | Proteins fetched: %s", reported, fetched)

    if protein_data.get("is_partial"):
        logger.warning("Output is incomplete — re-run the script to resume from the checkpoint.")

    check_count_anomaly(reported, fetched, max_abs=max_abs, max_percent=max_percent)

    output_data: ProteinResult = {
        "entry_accession": ENTRY_ACCESSION,
        "proteins_reported": reported,
        "total_proteins_fetched": fetched,
        "is_partial": protein_data.get("is_partial", False),
        "proteins": protein_data.get("proteins", []),
        "note": (
            "No deduplication - proteins are stored exactly as returned by the API. "
            "Duplicate accession values are possible only if the API returns duplicates."
        ),
    }

    await asyncio.to_thread(write_output, output_data, output_file)
    logger.info("Results written to %s", output_file)


def cli() -> None:
    """Console script entry point for fetch-proteins-dnak."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
