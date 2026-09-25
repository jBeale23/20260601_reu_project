"""Fetch functional annotation, transfer it by homology, and test the structural split."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp

from data_fetching.fetch_function import fetch_functions, write_function_store
from data_fetching.utils import INTERPRO_HEADERS
from domain_layout.records import load_domain_store
from validation.function_enrichment import SubpopulationPair, functional_split_report
from validation.function_transfer import (
    donor_sequences_from,
    experimental_donors,
    merge_observed_and_transferred,
    transfer_annotations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
WK = Path.home() / "scr4_sfried3/domain_layout_run"
STORE = WK / (sys.argv[1] if len(sys.argv) > 1 else "protein_domains_backfilled.json")
LAYOUT = WK / (sys.argv[2] if len(sys.argv) > 2 else "full_layout_v4")

# Inputs stay where they are; outputs are redirectable. The group quota on scr4_sfried3 is
# shared with the rest of the lab and has filled repeatedly, and when it is full every write
# here fails while reads keep working - so a run can spend a day computing and then have
# nowhere to put the answer. Reading from scratch and writing elsewhere keeps the run alive
# without moving anyone else's data.
OUT = Path(os.environ.get("TRANSFER_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

store = load_domain_store(STORE)
accessions = sorted(store.proteins)
print(f"store: {len(accessions)} proteins", flush=True)


async def main():
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as s:
        return await fetch_functions(s, accessions, concurrency=24, checkpoint=OUT / "function_store_checkpoint.json")


functions = asyncio.run(main())
write_function_store(functions, OUT / "function_store.json")
print(f"fetched {len(functions)} records, {len(functions.failures)} failures", flush=True)

recs = list(functions)
exp = [r for r in recs if r.has_experimental_annotation]
any_ann = [r for r in recs if r.has_any_annotation]
print(f"any annotation: {len(any_ann)} ({len(any_ann) / max(1, len(recs)):.1%})")
print(f"EXPERIMENTAL  : {len(exp)} ({len(exp) / max(1, len(recs)):.1%})", flush=True)

# Homology transfer from the experimentally annotated to everything else.
sequences = {a: store.proteins[a].sequence for a in accessions if store.proteins[a].sequence}
donors = experimental_donors(functions.records, sequences)
donor_seqs = donor_sequences_from(functions.records, sequences)
targets = {a: s for a, s in sequences.items() if a not in donors}
print(f"donors: {len(donors)}   targets: {len(targets)}", flush=True)

transferred, summary = transfer_annotations(targets, donors, donor_seqs)
print("transfer:", json.dumps(summary.to_json_dict(), indent=2), flush=True)
(OUT / "function_transfer.json").write_text(
    json.dumps({a: t.to_json_dict() for a, t in transferred.items()}, indent=2) + "\n"
)

# The experiment: does the structural split predict function?
import csv

sub = {}
with (LAYOUT / "domain_layout_features.csv").open() as fh:
    for row in csv.DictReader(fh):
        sub[row["accession"]] = row.get("layout_predicted_subclass", "")
groups = {}
for a, s in sub.items():
    groups.setdefault(s, []).append(a)

merged_terms = merge_observed_and_transferred(functions.records, transferred)
print(f"\nproteins with any usable term (observed or transferred): {len(merged_terms)}", flush=True)

out = {}
for x, y in (("b_gf_rich_no_ctd", "b_canonical"), ("c_j_domain_only", "c_atypical_multi_domain")):
    if x not in groups or y not in groups:
        continue
    rep = functional_split_report(SubpopulationPair(x, y, groups[x], groups[y]), functions.records)
    out[f"{x}_vs_{y}"] = rep
    print(f"\n=== {x} vs {y} ===")
    print(
        f"  coverage: {rep['coverage'][x]['annotation_coverage']:.1%} vs {rep['coverage'][y]['annotation_coverage']:.1%}"
    )
    print(f"  terms tested: {rep['n_terms_tested']}   significant: {rep['n_significant']}")
    for t in rep["top_terms"][:8]:
        print(
            f"    {t['source']:22s} {t['term'][:44]:44s} {t['rate_in_group']:.3f} vs {t['rate_in_other']:.3f}  "
            f"OR={t['odds_ratio']:7.2f}  p_adj={t['p_adjusted']:.2e}"
        )
(OUT / "function_enrichment.json").write_text(json.dumps(out, indent=2) + "\n")
print("\nwrote function_store.json, function_transfer.json, function_enrichment.json")
