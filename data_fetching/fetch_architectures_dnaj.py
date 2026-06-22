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
                "is_partial": False,
                "proteins": [{"appears_in_architecture_count": 2}],
            },
        ],
        "total_proteins_fetched": 225610,
        "note": "No deduplication - proteins may appear multiple times...",
    }
"""

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
from tqdm import tqdm

from data_fetching.fetch_types import ApiResponse, ArchResult, ProteinCounts
from data_fetching.utils import (
    INTERPRO_HEADERS,
    check_count_anomaly,
    configure_logging,
    get_with_retry,
    validate_api_response,
    write_output,
)

configure_logging()
logger = logging.getLogger(__name__)

# --- Constants ---
_DEFAULT_ARCH_URL = "https://www.ebi.ac.uk/interpro/api/entry/InterPro/IPR001623/?ida&page_size=20&format=json"
_DEFAULT_N_ARCHITECTURES = 20
_DEFAULT_CONCURRENCY = 5
_DEFAULT_OUTPUT = Path("ipr001623_domain_architectures_no_dedup.json")

PROTEIN_URL_TEMPLATE = (
    "https://www.ebi.ac.uk/interpro/api/protein/UniProt/"
    "entry/InterPro/IPR001623/?ida={ida_id}&page_size=200&format=json"
)

# Known values from the InterPro website, used to confirm we start at architecture #1
EXPECTED_FIRST_IDA_ID = "500088c3adc88e8af670fe08554083396acf46f3"
EXPECTED_FIRST_PROTEIN_COUNT = 98256


async def fetch_proteins_for_arch(
    session: aiohttp.ClientSession,
    arch: ApiResponse,
    index: int,
    semaphore: asyncio.Semaphore,
    n_architectures: int,
) -> ArchResult:
    """Fetch all proteins for a single domain architecture via cursor pagination.

    Pages must be fetched sequentially because each page's URL is returned
    by the previous response. No deduplication is performed. If retries are
    exhausted mid-pagination, the partial results collected so far are returned
    rather than crashing the entire run. In that case ``is_partial`` is set to
    ``True`` in the returned dict so callers can distinguish a truncated result
    from a complete one.

    Args:
        session: The aiohttp client session to use.
        arch: Architecture metadata dict from the InterPro API.
        index: 1-based index of this architecture in the run (for display).
        semaphore: Shared semaphore limiting concurrent architecture fetches.
        n_architectures: Total number of architectures being fetched (for display).

    Returns:
        Dictionary containing architecture metadata, all fetched proteins, and
        an ``is_partial`` flag that is ``True`` only if pagination was cut short
        by retry exhaustion.
    """
    ida_id: str = arch.get("ida_id", "")
    proteins: list[ApiResponse] = []
    url: str | None = PROTEIN_URL_TEMPLATE.format(ida_id=ida_id)
    is_partial = False

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
                    logger.exception(
                        "Retries exhausted mid-pagination for arch %s (%s), returning %s proteins collected so far",
                        index,
                        ida_id,
                        len(proteins),
                    )
                    is_partial = True
                    break
                batch: list[ApiResponse] = data.get("results", [])
                proteins.extend(batch)
                pbar.update(len(batch))
                url = data.get("next")

    return {
        "ida": arch.get("ida", ""),
        "ida_id": ida_id,
        "unique_proteins_reported": arch.get("unique_proteins", 0),
        "proteins_fetched": len(proteins),
        "is_partial": is_partial,
        "proteins": proteins,
    }


async def main() -> None:
    """Fetch all proteins for the top N domain architectures of IPR001623."""
    parser = argparse.ArgumentParser(description="Fetch IPR001623 domain architectures from InterPro.")
    parser.add_argument(
        "-n",
        "--n-architectures",
        type=int,
        default=None,
        help=f"Number of architectures to fetch (default: {_DEFAULT_N_ARCHITECTURES})",
    )
    parser.add_argument("-u", "--arch-url", type=str, default=None, help="Architecture list URL")
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=None,
        help=f"Max concurrent requests (default: {_DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help=f"Output JSON file path (default: {_DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    n_architectures = args.n_architectures if args.n_architectures is not None else _DEFAULT_N_ARCHITECTURES
    arch_url = args.arch_url if args.arch_url is not None else _DEFAULT_ARCH_URL
    concurrency = args.concurrency if args.concurrency is not None else _DEFAULT_CONCURRENCY
    output_file = args.output if args.output is not None else _DEFAULT_OUTPUT

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS) as session:
        logger.info("Fetching architecture groups from InterPro...")
        arch_data = await get_with_retry(session, arch_url)
        validate_api_response(arch_data)
        architectures: list[ApiResponse] = arch_data.get("results", [])[:n_architectures]
        logger.info("Got %s architecture group(s).", len(architectures))

        logger.info("Fetching proteins...")
        tasks = [
            fetch_proteins_for_arch(session, arch, i, semaphore, n_architectures)
            for i, arch in enumerate(architectures, start=1)
        ]
        all_results: tuple[ArchResult, ...] = await asyncio.gather(*tasks)

    logger.info("Processing duplicate flags...")
    protein_occurrence_count: ProteinCounts = {}

    for result in all_results:
        for protein in result["proteins"]:
            accession: str | None = protein.get("metadata", {}).get("accession")
            if accession:
                protein_occurrence_count[accession] = protein_occurrence_count.get(accession, 0) + 1

    for result in all_results:
        for protein in result["proteins"]:
            accession = protein.get("metadata", {}).get("accession")
            if accession:
                protein["appears_in_architecture_count"] = protein_occurrence_count[accession]

    partial_archs = [r for r in all_results if r.get("is_partial")]
    if partial_archs:
        logger.warning(
            "%s architecture(s) returned incomplete results due to retry exhaustion: %s",
            len(partial_archs),
            [r["ida_id"] for r in partial_archs],
        )

    for result in all_results:
        check_count_anomaly(
            result["unique_proteins_reported"],
            result["proteins_fetched"],
        )

    total_fetched = sum(r["proteins_fetched"] for r in all_results)

    output_data: ArchResult = {
        "architectures": list(all_results),
        "total_proteins_fetched": total_fetched,
        "note": (
            "No deduplication - proteins may appear multiple times if present in "
            "multiple architectures. Use 'appears_in_architecture_count' to identify duplicates."
        ),
    }

    await asyncio.to_thread(write_output, output_data, output_file)
    logger.info("Results written to %s", output_file)


def cli() -> None:
    """Console script entry point for fetch-architectures-dnaj."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
