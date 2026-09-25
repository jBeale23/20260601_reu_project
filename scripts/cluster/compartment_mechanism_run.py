"""Is subcellular compartment the mechanism linking architecture to Hsp70 choice?

The within-kingdom result showed that eukaryotic domain architecture predicts which Hsp70
paralogue a J-domain protein partners, and that the groups look compartment-specific: a
J-domain preceded by another domain goes with mitochondrial HSPA9, a J-domain followed by
others with the ER's HSPA5. But "looks compartment-specific" was read off the partner names,
which is an interpretation, not a measurement.

This measures it. UniProt's curated subcellular location comments are an independent
statement of where a protein goes - curated from experiments, and in no way derived from the
order of its domains. Testing the chain link by link is what turns the interpretation into a
claim that could fail:

1. Does architecture predict the *annotated compartment*?
2. Does the annotated compartment predict the Hsp70 partner?
3. Does architecture still predict the partner once compartment is held fixed?

A mechanism running architecture -> compartment -> partner predicts yes, yes, and *no*. If
architecture keeps predicting the partner within a single compartment, then compartment is
not the whole story and something else is carrying the signal.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_string import CURATED_CHANNELS, Partner, StringStore
from data_fetching.fetch_taxonomy import EUKARYOTA
from validation.pairing import PARALOGUE, classify_partner
from validation.partition_search import candidate_partitions
from validation.partner_prediction import predict_out_of_fold
from validation.partner_readouts import dominant_label

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
TRANSFER = Path("/home/jbeale3/transfer_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

SCHEME = "top_architectures"
MIN_COMPARTMENT = 40

# UniProt location strings are free text at several granularities - "Mitochondrion inner
# membrane", "Mitochondrion matrix" - so they are folded to the compartment that determines
# which Hsp70 pool a protein meets. Order matters: the first match wins, and the organelles
# are checked before the generic "Membrane" and "Cytoplasm" that often accompany them.
COMPARTMENT_RULES = (
    ("mitochondrion", "mitochondrion"),
    ("endoplasmic reticulum", "endoplasmic_reticulum"),
    ("chloroplast", "chloroplast"),
    ("peroxisome", "peroxisome"),
    ("nucleus", "nucleus"),
    ("cytosol", "cytosol"),
    ("cytoplasm", "cytosol"),
)


def compartment_of(locations) -> str | None:
    """Fold UniProt location strings to one compartment, or None if none applies."""
    lowered = [str(item).lower() for item in locations]
    for needle, name in COMPARTMENT_RULES:
        if any(needle in item for item in lowered):
            return name
    return None


def main():
    features, organism = {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if accession:
                features[accession] = row
                organism[accession] = (row.get("organism_name") or "").strip()

    store = StringStore()
    for accession, rows in json.loads((OUT / "string_store.json").read_text())["partners"].items():
        store.partners[accession] = [Partner.from_json_dict(r) for r in rows]
    partner = dominant_label(store.names_by_accession(CURATED_CHANNELS), lambda g: classify_partner(g, PARALOGUE))

    taxonomy = json.loads((OUT / "taxonomy_store.json").read_text())["superkingdom"]
    eukaryotes = {a for a in partner if taxonomy.get(organism.get(a, ""), "") == EUKARYOTA}

    records = json.loads((TRANSFER / "function_store.json").read_text())["records"]
    compartment = {}
    for accession in eukaryotes:
        name = compartment_of(records.get(accession, {}).get("subcellular_locations", ()))
        if name:
            compartment[accession] = name

    print(f"eukaryotic proteins with a partner: {len(eukaryotes):,}", flush=True)
    print(f"  of those with a curated compartment: {len(compartment):,}", flush=True)
    print(f"  {dict(Counter(compartment.values()).most_common())}", flush=True)

    candidates = candidate_partitions(features)
    partition = candidates[SCHEME]
    genus = {a: (organism[a].split()[0] if organism[a] else "unknown") for a in organism}
    report = {"n_eukaryotes": len(eukaryotes), "n_with_compartment": len(compartment)}

    # Link 1: architecture -> compartment.
    print(f"\n{'=' * 92}\nLINK 1: does architecture predict the annotated compartment?\n{'=' * 92}")
    link1 = predict_out_of_fold(
        {a: partition[a] for a in compartment if a in partition},
        compartment,
        genus,
        scheme=f"{SCHEME}->compartment",
    )
    print(
        f"  accuracy {link1.accuracy:.1%}  majority {link1.majority_accuracy:.1%}  "
        f"lift {link1.lift_over_majority:+.1%}  n={link1.n_predicted}"
    )
    report["architecture_to_compartment"] = link1.to_json_dict()

    # Link 2: compartment -> partner.
    print(f"\n{'=' * 92}\nLINK 2: does the compartment predict the Hsp70 partner?\n{'=' * 92}")
    link2 = predict_out_of_fold(
        compartment,
        {a: partner[a] for a in compartment},
        genus,
        scheme="compartment->partner",
    )
    print(
        f"  accuracy {link2.accuracy:.1%}  majority {link2.majority_accuracy:.1%}  "
        f"lift {link2.lift_over_majority:+.1%}  n={link2.n_predicted}"
    )
    report["compartment_to_partner"] = link2.to_json_dict()

    for name, members in sorted(
        ((c, [a for a in compartment if compartment[a] == c]) for c in set(compartment.values())),
        key=lambda kv: -len(kv[1]),
    ):
        counts = Counter(partner[a] for a in members)
        top, hits = counts.most_common(1)[0]
        print(f"    {name:24s} n={len(members):4d}  {top:8s} {hits / len(members):.0%}  {dict(counts.most_common(3))}")

    # Link 3: architecture -> partner, holding compartment fixed.
    print(f"\n{'=' * 92}\nLINK 3: does architecture still predict the partner within one compartment?\n{'=' * 92}")
    within = {}
    for name in sorted(set(compartment.values())):
        members = [a for a in compartment if compartment[a] == name and a in partition]
        if len(members) < MIN_COMPARTMENT:
            continue
        result = predict_out_of_fold(
            {a: partition[a] for a in members},
            {a: partner[a] for a in members},
            genus,
            scheme=f"{SCHEME}|{name}",
        )
        within[name] = result.to_json_dict()
        print(
            f"  {name:24s} n={result.n_predicted:4d}  accuracy {result.accuracy:.1%}  "
            f"majority {result.majority_accuracy:.1%}  lift {result.lift_over_majority:+.1%}"
        )
    report["within_compartment"] = within

    lifts = [v["lift_over_majority"] for v in within.values()]
    if lifts:
        mean_lift = sum(lifts) / len(lifts)
        print(f"\n  mean lift within compartments: {mean_lift:+.1%}")
        print(
            "  A mechanism running architecture -> compartment -> partner predicts this "
            "collapses toward zero.\n  A lift that survives means architecture carries "
            "something compartment does not."
        )
        report["mean_within_compartment_lift"] = round(mean_lift, 4)

    (OUT / "compartment_mechanism.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT / 'compartment_mechanism.json'}")


if __name__ == "__main__":
    main()
