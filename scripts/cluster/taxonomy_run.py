"""Resolve organisms to a superkingdom, for the proteins that have a partner readout.

The full store spans 22,076 organisms, and resolving all of them would take hours to answer a
question that only concerns the proteins carrying an Hsp70 partner. Restricting to those cuts
the work about sixfold for the same answer. Anything already resolved is reused from the
checkpoint.
"""

import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_string import CURATED_CHANNELS, Partner, StringStore
from data_fetching.fetch_taxonomy import fetch_taxonomy, write_taxonomy_store
from data_fetching.utils import INTERPRO_HEADERS
from validation.pairing import PARALOGUE, classify_partner
from validation.partner_readouts import dominant_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

store = StringStore()
payload = json.loads((OUT / "string_store.json").read_text())
for accession, rows in payload["partners"].items():
    store.partners[accession] = [Partner.from_json_dict(r) for r in rows]
readout = set(dominant_label(store.names_by_accession(CURATED_CHANNELS), lambda g: classify_partner(g, PARALOGUE)))
print(f"proteins with a partner readout: {len(readout):,}", flush=True)

organisms = set()
with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
    for row in csv.DictReader(fh):
        name = (row.get("organism_name") or "").strip()
        if name and (row.get("accession") or "").strip() in readout:
            organisms.add(name)
print(f"organisms to resolve: {len(organisms):,}", flush=True)


async def main():
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
        return await fetch_taxonomy(session, sorted(organisms), checkpoint=OUT / "taxonomy_checkpoint.json")


store = asyncio.run(main())
write_taxonomy_store(store, OUT / "taxonomy_store.json")
print(f"\nresolved {len(store):,} organisms, {len(store.failures):,} failures")
print(f"  {store.counts()}")
print(f"wrote {OUT / 'taxonomy_store.json'}")
