"""Fetch STRING interaction partners for every J-domain protein.

BioGRID reaches 1,089 of 217,701, and 677 after requiring an unambiguous Hsp70 partner.
That number is the binding constraint on every partner-based result in the project. STRING
covers thousands of genomes rather than four model organisms, and a probe put the yield at
28% of the proteins BioGRID misses.
"""

import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

import aiohttp

from data_fetching.fetch_string import (
    CONTEXT_CHANNELS,
    CURATED_CHANNELS,
    fetch_string_partners,
    write_string_store,
)
from data_fetching.utils import INTERPRO_HEADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

# The shared fetch helper logs the full body of every 404. For most APIs that is the right
# default; for this one a 404 means "STRING has no entry for this protein", which is the
# common case here - roughly 150,000 of them, each carrying a paragraph of HTML. Left at
# ERROR the run would write about 100 MB of stderr saying nothing. The outcome is still
# recorded per accession in the store.
logging.getLogger("data_fetching.utils").setLevel(logging.CRITICAL)

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

accessions = []
with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
    for row in csv.DictReader(fh):
        accession = (row.get("accession") or "").strip()
        if accession:
            accessions.append(accession)
print(f"accessions to fetch: {len(accessions):,}", flush=True)


async def main():
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
        return await fetch_string_partners(
            session,
            accessions,
            checkpoint=OUT / "string_checkpoint.json",
        )


store = asyncio.run(main())
write_string_store(store, OUT / "string_store.json")

curated = store.names_by_accession(CURATED_CHANNELS)
context = store.names_by_accession(CONTEXT_CHANNELS)
print(
    f"\nfetched {len(store):,} with a STRING entry; "
    f"{store.n_absent:,} absent from STRING; {store.n_failed:,} genuine failures"
)
print(f"  with curated-evidence partners : {len(curated):,}")
print(f"  with genomic-context partners  : {len(context):,}")
print(f"  with either                    : {len(set(curated) | set(context)):,}")

# The pair the whole system is named for: how often is a DnaK-family chaperone a partner?
dnak_like = {"DNAK", "HSCA", "HSCC"}
with_dnak = sum(1 for names in curated.values() if any(n.upper() in dnak_like for n in names))
print(f"  carrying a DnaK-family partner : {with_dnak:,}")
print(f"\nwrote {OUT / 'string_store.json'}")
