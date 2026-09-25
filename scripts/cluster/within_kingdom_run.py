"""Does domain architecture predict Hsp70 choice *within* a kingdom?

The cross-kingdom result is trivial and misleading. A bacterial protein partners DnaK because
DnaK is the Hsp70 its genome encodes, and its architecture is bacterial for the same
phylogenetic reason - so a partition that separates bacterial from eukaryotic architectures
will look like it predicts partner identity while predicting only taxonomy.

Inside a kingdom that shortcut is gone. A eukaryotic cell carries HSPA5 in the ER, HSPA8 in
the cytosol and HSPA9 in mitochondria at once, so which one a J-domain protein partners is a
real choice rather than a consequence of which genes the organism has. If the purity gradient
survives here it is a mechanism; if it vanishes, the partition result was phylogeny.

The blocking ladder runs alongside. Species and genus are not independent controls - species
nest inside genera, so blocking genus already blocks species - but running both shows how much
leakage each level was carrying.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_string import CURATED_CHANNELS, Partner, StringStore
from data_fetching.fetch_taxonomy import BACTERIA, EUKARYOTA
from validation.pairing import PARALOGUE, classify_partner
from validation.partition_search import candidate_partitions
from validation.partner_prediction import group_partner_profile, predict_out_of_fold
from validation.partner_readouts import dominant_label

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

WINNERS = [
    "top_architectures",
    "idr_tertile+n_structured_domains",
    "has_dnaj_c+idr_tertile",
    "has_dnaj_c+n_structured_domains",
    "has_gf_rich_region+n_structured_domains",
]

MIN_KINGDOM = 300


def main():
    features, reference, organism = {}, {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            features[accession] = row
            organism[accession] = (row.get("organism_name") or "").strip()
            layout_class = (row.get("layout_predicted_class") or "").strip()
            if layout_class in {"A", "B", "C"}:
                reference[accession] = layout_class

    taxonomy = json.loads((OUT / "taxonomy_store.json").read_text())["superkingdom"]
    kingdom = {a: taxonomy.get(o, "unknown") for a, o in organism.items()}

    store = StringStore()
    payload = json.loads((OUT / "string_store.json").read_text())
    for accession, rows in payload["partners"].items():
        store.partners[accession] = [Partner.from_json_dict(r) for r in rows]
    partner = dominant_label(store.names_by_accession(CURATED_CHANNELS), lambda g: classify_partner(g, PARALOGUE))

    scored = [a for a in partner if a in kingdom]
    print(f"readout: {len(partner):,}   with a kingdom: {len(scored):,}", flush=True)
    print(f"  {dict(Counter(kingdom[a] for a in scored).most_common())}", flush=True)

    candidates = candidate_partitions(features)
    schemes = {"A/B/C": reference} | {n: candidates[n] for n in WINNERS if n in candidates}

    species = {a: " ".join(organism[a].split()[:2]) for a in organism}
    genus = {a: (organism[a].split()[0] if organism[a] else "unknown") for a in organism}
    blockings = {"species": species, "genus": genus}

    report = {}
    for realm in (BACTERIA, EUKARYOTA):
        members = {a for a in scored if kingdom[a] == realm}
        if len(members) < MIN_KINGDOM:
            print(f"\n{realm}: only {len(members):,} proteins, below the floor of {MIN_KINGDOM}")
            continue
        realm_partner = {a: partner[a] for a in members}
        top = Counter(realm_partner.values()).most_common(5)
        print(f"\n{'=' * 96}\n{realm.upper()} - n = {len(members):,}\n{'=' * 96}")
        print(f"  partners present: {dict(top)}")

        print(f"\n  {'scheme':42s} {'block':>9s} {'n_pred':>8s} {'accuracy':>9s} {'majority':>9s} {'lift':>8s}")
        rows = []
        for name, partition in schemes.items():
            for block_name, block in blockings.items():
                result = predict_out_of_fold(
                    {a: partition[a] for a in members if a in partition},
                    realm_partner,
                    block,
                    scheme=f"{name}|{block_name}",
                )
                rows.append(result.to_json_dict() | {"kingdom": realm, "blocking": block_name})
                print(
                    "  %-42s %9s %8d %8.1f%% %8.1f%% %+7.1f%%"
                    % (
                        name[:41],
                        block_name,
                        result.n_predicted,
                        100 * result.accuracy,
                        100 * result.majority_accuracy,
                        100 * result.lift_over_majority,
                    )
                )
        report[realm] = rows

        # The purity gradient, inside this kingdom only.
        best = max(rows, key=lambda r: r["lift_over_majority"])
        scheme_name = best["scheme"].split("|")[0]
        profile = group_partner_profile(
            {a: schemes[scheme_name][a] for a in members if a in schemes[scheme_name]},
            realm_partner,
            min_members=20,
        )
        report[f"{realm}_profile"] = profile
        print(f"\n  purity within {realm}, best scheme ({scheme_name}):")
        for group, data in sorted(profile.items(), key=lambda kv: -kv[1]["purity"]):
            print(
                f"    {group[:40]:42s} n={data['n']:5d}  {data['dominant_partner']:10s} "
                f"purity={data['purity']:.0%}  {data['top_partners']}"
            )

    (OUT / "within_kingdom.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT / 'within_kingdom.json'}")


if __name__ == "__main__":
    main()
