"""Fetch evidence-coded GO annotation for the whole store, then report the tier yield.

The UniProt route finds 198 proteins with experimental terms because UniProt exposes
evidence records only for reviewed entries. QuickGO serves the GO consortium's own store and
sampling put the experimental yield at roughly 2.7x that, with a much larger phylogenetic
tier on top.
"""

import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

import aiohttp

from data_fetching.fetch_go_evidence import (
    TIER_CURATED,
    TIER_EXPERIMENTAL,
    TIER_PHYLOGENETIC,
    fetch_go_evidence,
    write_evidence_store,
)
from data_fetching.utils import INTERPRO_HEADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)
LAYOUT = WK / "full_layout_v6" / "domain_layout_features.csv"

accessions = []
with LAYOUT.open() as fh:
    for row in csv.DictReader(fh):
        accession = (row.get("accession") or "").strip()
        if accession:
            accessions.append(accession)
print(f"accessions to fetch: {len(accessions)}", flush=True)


async def main():
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
        return await fetch_go_evidence(
            session,
            accessions,
            concurrency=16,
            checkpoint=OUT / "go_evidence_checkpoint.json",
        )


store = asyncio.run(main())
write_evidence_store(store, OUT / "go_evidence_store.json")

counts = store.tier_counts()
print(f"\nfetched {len(store)} records, {len(store.failures)} failures", flush=True)
print("proteins carrying at least one term per tier:")
for tier, count in counts.items():
    print(f"  {tier:14s} {count:7,}  ({count / max(1, len(store)):.2%})")

for tier in (TIER_EXPERIMENTAL, TIER_PHYLOGENETIC, TIER_CURATED):
    cumulative = store.terms_by_accession(tier, cumulative=True)
    print(f"  at or above {tier:14s}: {len(cumulative):7,} proteins")

print(f"\nwrote {OUT / 'go_evidence_store.json'}")
