"""Predict every protein's Hsp70 partner from its group, and check whether it is right.

The association tests said the new schemes track partner identity. This asks whether they can
be used: assign each protein the commonest partner among the other members of its group, and
score that against the partner it actually has.

Predictions are out of fold and folds are blocked by genus, so no protein is predicted by its
own near-identical relatives. That is what makes the number an estimate of performance on an
organism nobody has studied, which is what the project is for. The baseline is the majority
partner, computed the same way.

The second half is the part that could yield a mechanism. An accuracy says a scheme predicts;
it does not say what it predicts. Reading off which group goes with which chaperone, and how
purely, turns a statistical result into a set of specific claims that could be wrong.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_string import CONTEXT_CHANNELS, CURATED_CHANNELS, Partner, StringStore
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
    "idr_tertile+j_domain_position",
    "has_dnaj_c+idr_tertile",
    "has_dnaj_c+n_structured_domains",
    "has_gf_rich_region+idr_tertile",
    "has_zinc_finger_like+idr_tertile",
    "has_gf_rich_region+j_domain_position",
    "has_gf_rich_region+n_structured_domains",
    "has_dnaj_c+has_gf_rich_region",
]


def main():
    features, reference, genus = {}, {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            features[accession] = row
            organism = (row.get("organism_name") or "").strip()
            genus[accession] = organism.split()[0] if organism else "unknown"
            layout_class = (row.get("layout_predicted_class") or "").strip()
            if layout_class in {"A", "B", "C"}:
                reference[accession] = layout_class

    store = StringStore()
    payload = json.loads((OUT / "string_store.json").read_text())
    for accession, rows in payload["partners"].items():
        store.partners[accession] = [Partner.from_json_dict(r) for r in rows]

    readouts = {
        "curated": dominant_label(store.names_by_accession(CURATED_CHANNELS), lambda g: classify_partner(g, PARALOGUE)),
        "context": dominant_label(store.names_by_accession(CONTEXT_CHANNELS), lambda g: classify_partner(g, PARALOGUE)),
    }
    candidates = candidate_partitions(features)
    schemes = {"A/B/C": reference} | {n: candidates[n] for n in WINNERS if n in candidates}

    report = {}
    for readout_name, partner in readouts.items():
        print(f"\n{'=' * 96}\nPREDICTING PARTNER FROM GROUP - {readout_name} channel, n = {len(partner):,}\n{'=' * 96}")
        print("  %-44s %6s %8s %8s %9s %9s %8s" % ("scheme", "grps", "n_pred", "cover", "accuracy", "majority", "lift"))
        rows = []
        for name, partition in schemes.items():
            result = predict_out_of_fold(partition, partner, genus, scheme=name)
            rows.append(result)
            print(
                "  %-44s %6d %8d %8.1f%% %8.1f%% %8.1f%% %+7.1f%%"
                % (
                    name[:43],
                    result.n_groups,
                    result.n_predicted,
                    100 * result.coverage,
                    100 * result.accuracy,
                    100 * result.majority_accuracy,
                    100 * result.lift_over_majority,
                )
            )
        report[readout_name] = [r.to_json_dict() for r in rows]

    # Where a mechanism would show: which group goes with which chaperone, and how purely.
    partner = readouts["curated"]
    print(f"\n{'=' * 96}\nWHAT EACH GROUP PREDICTS - curated channel\n{'=' * 96}")
    profiles = {}
    best = max(
        (r for r in report["curated"] if r["scheme"] != "A/B/C"),
        key=lambda r: r["lift_over_majority"],
    )
    for name in ("A/B/C", best["scheme"]):
        partition = schemes[name]
        profile = group_partner_profile(partition, partner, min_members=20)
        profiles[name] = profile
        print(f"\n  {name}")
        for group, data in sorted(profile.items(), key=lambda kv: -kv[1]["purity"]):
            print(
                f"    {group[:40]:42s} n={data['n']:5d}  {data['dominant_partner']:10s} "
                f"purity={data['purity']:.0%}  {data['top_partners']}"
            )
    report["group_profiles"] = profiles
    report["most_predictive_scheme"] = best["scheme"]

    (OUT / "partner_prediction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT / 'partner_prediction.json'}")


if __name__ == "__main__":
    main()
