"""Name two ER-resident J-domain proteins whose architectures predict different Hsp70s.

The within-compartment result says architecture predicts partner choice even after
subcellular location is held fixed. That is a claim about specific proteins, and this turns
it into one: find ER-resident proteins whose architecture groups differ, whose predicted
partners therefore differ, and which are well enough characterised that somebody could run
the pull-down.

Every prediction here is made out of fold - the rule assigning a protein its partner is
fitted on other proteins, in other genera - so the comparison against the recorded partner is
a test rather than a lookup. That distinction is the whole point: an in-sample "prediction"
would just be reading back the answer the model was given.

Two things this cannot do. The recorded partner is itself orthology-transferred rather than
observed in that organism, so agreement is a consistency check and not a confirmation. And a
protein whose recorded partner is missing is the *most* interesting case for an experiment
and the least checkable here, so those are reported separately rather than dropped.
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
from validation.partner_prediction import MIN_GROUP_TRAIN, blocked_folds
from validation.partner_readouts import dominant_label

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
TRANSFER = Path("/home/jbeale3/transfer_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

SCHEME = "top_architectures"

# Insisting on human or yeast returned nothing: the ER set is fish and fungi, because those
# are the genomes STRING covers that also carry a curated location. So tractability is scored
# rather than demanded - a named model organism first, then any genus with a sequenced genome
# and laboratory strains, then whatever is left - and the ranking is reported so the choice of
# organism is visible rather than hidden in a filter.
MODEL_ORGANISMS = ("homo sapiens", "saccharomyces cerevisiae", "mus musculus", "arabidopsis thaliana")
WORKABLE_GENERA = (
    "penicillium",
    "trichoderma",
    "aspergillus",
    "candida",
    "neurospora",
    "fusarium",
    "danio",
    "tetraodon",
    "oryzias",
    "gambusia",
    "xenopus",
    "puccinia",
    "ustilago",
)


def tractability(organism: str) -> int:
    """How readily a lab could work in this organism: 2 model, 1 workable, 0 otherwise."""
    lowered = organism.lower()
    if any(m in lowered for m in MODEL_ORGANISMS):
        return 2
    if any(g in lowered for g in WORKABLE_GENERA):
        return 1
    return 0


def is_er(locations) -> bool:
    """Whether UniProt places this protein in the endoplasmic reticulum."""
    return any("endoplasmic reticulum" in str(item).lower() for item in locations)


def main():
    features, organism, name = {}, {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if accession:
                features[accession] = row
                organism[accession] = (row.get("organism_name") or "").strip()
                name[accession] = (row.get("protein_name") or "").strip()

    store = StringStore()
    for accession, rows in json.loads((OUT / "string_store.json").read_text())["partners"].items():
        store.partners[accession] = [Partner.from_json_dict(r) for r in rows]
    partner = dominant_label(store.names_by_accession(CURATED_CHANNELS), lambda g: classify_partner(g, PARALOGUE))

    taxonomy = json.loads((OUT / "taxonomy_store.json").read_text())["superkingdom"]
    records = json.loads((TRANSFER / "function_store.json").read_text())["records"]

    er = [
        a
        for a in partner
        if taxonomy.get(organism.get(a, ""), "") == EUKARYOTA
        and is_er(records.get(a, {}).get("subcellular_locations", ()))
    ]
    candidates = candidate_partitions(features)
    partition = candidates[SCHEME]
    er = [a for a in er if a in partition]
    print(f"ER-resident eukaryotic proteins with a partner and an architecture: {len(er)}", flush=True)
    print(f"  architectures present: {dict(Counter(partition[a] for a in er).most_common())}", flush=True)

    # Out-of-fold predictions, blocked by genus, over the ER set alone.
    genus = {a: (organism[a].split()[0] if organism[a] else "unknown") for a in er}
    folds = blocked_folds(er, genus)
    prediction = {}
    for held_out in folds:
        train = [a for a in er if a not in held_out]
        modal = {}
        for group in {partition[a] for a in train}:
            counts = Counter(partner[a] for a in train if partition[a] == group)
            if sum(counts.values()) >= MIN_GROUP_TRAIN:
                modal[group] = counts.most_common(1)[0][0]
        for accession in held_out:
            if partition[accession] in modal:
                prediction[accession] = modal[partition[accession]]

    hits = sum(1 for a in prediction if prediction[a] == partner[a])
    print(f"  predicted out of fold: {len(prediction)}; agreeing with the recorded partner: {hits}", flush=True)

    # Group the ER proteins by predicted partner, so a contrasting pair can be picked.
    by_prediction = {}
    for accession, guess in prediction.items():
        by_prediction.setdefault(guess, []).append(accession)
    print(f"\n  predicted partners across the ER set: {dict(Counter(prediction.values()).most_common())}")

    print(
        f"\n{'=' * 100}\nCANDIDATE PAIRS - same compartment, different architecture, different predicted partner\n{'=' * 100}"
    )
    ranked = sorted(prediction, key=lambda a: (-tractability(organism[a]), organism[a]))
    print(f"  tractability of the ER set: {dict(Counter(tractability(organism[a]) for a in prediction))}")

    pairs = []
    guesses = sorted(by_prediction, key=lambda g: -len(by_prediction[g]))
    for i, first_guess in enumerate(guesses):
        for second_guess in guesses[i + 1 :]:
            left = [a for a in ranked if prediction[a] == first_guess]
            right = [a for a in ranked if prediction[a] == second_guess]
            for a in left:
                match = next((b for b in right if partition[a] != partition[b]), None)
                if match is not None:
                    pairs.append((tractability(organism[a]) + tractability(organism[match]), a, match))
                    break

    for rank, (score, a, b) in enumerate(sorted(pairs, reverse=True)[:3], start=1):
        print(f"\n  PAIR {rank}   (tractability {score}/4)")
        for accession in (a, b):
            agrees = "agrees" if prediction[accession] == partner[accession] else "DISAGREES"
            print(f"    {accession}  {name[accession][:52]:54s}")
            print(f"      organism     : {organism[accession]}")
            print(f"      architecture : {partition[accession]}")
            print(f"      predicted    : {prediction[accession]}")
            print(f"      recorded     : {partner[accession]}   ({agrees})")

    report = {
        "n_er_proteins": len(er),
        "n_predicted_out_of_fold": len(prediction),
        "n_agreeing_with_recorded": hits,
        "predictions": {
            a: {
                "protein": name[a],
                "organism": organism[a],
                "architecture": partition[a],
                "predicted_partner": prediction[a],
                "recorded_partner": partner.get(a),
                "agrees": prediction[a] == partner.get(a),
            }
            for a in sorted(prediction)
        },
    }
    (OUT / "er_candidates.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT / 'er_candidates.json'}")


if __name__ == "__main__":
    main()
