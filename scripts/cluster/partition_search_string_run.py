"""Re-run the partition search on STRING partners rather than BioGRID.

Every partner-based result so far rests on the 677 proteins BioGRID reaches with an
unambiguous Hsp70 partner. That is the binding constraint, and it is a coverage limit rather
than a statistical one: BioGRID curates the literature, and the literature is four model
organisms. STRING covers thousands of genomes, most of them bacterial, which is where DnaJ
and DnaK actually live.

The readouts are built from separate evidence channels and never pooled:

``curated``
    STRING's experimental and database channels. For a bacterium these are usually
    transferred from a studied relative by whole-protein orthology - inference, but not
    inference from domain architecture, which is what would make it circular here.
``context``
    Neighborhood, fusion, and co-occurrence: genes that sit together, fuse, or appear and
    vanish together across genomes. Available for any sequenced genome and independent of
    both structure and curation - the most orthogonal evidence at this scale.

Which channel supplies the discovery readout is set by ``STRING_DISCOVERY_CHANNEL``. Running
it both ways is the point: ``curated`` and ``context`` rest on entirely different evidence -
one on experiments transferred by orthology, the other on where genes sit in genomes - so a
partition that wins under both has replicated on independent data rather than been confirmed
on a correlated readout. Whichever channel ranks, the other one validates.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_string import (
    CONTEXT_CHANNELS,
    CURATED_CHANNELS,
    Partner,
    StringStore,
)
from validation.pairing import PARALOGUE, classify_partner, is_hsp70
from validation.partition_search import (
    CRITERION_INFORMATION,
    CRITERION_V,
    candidate_partitions,
    search_partitions,
)
from validation.partner_readouts import (
    describe_relationship,
    dominant_label,
    interactome_scope,
    partition_relationship,
    tie_rate,
)
from validation.partner_systems import classify_non_hsp70

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    features, reference = {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            layout_class = (row.get("layout_predicted_class") or "").strip()
            if accession and layout_class in {"A", "B", "C"}:
                reference[accession] = layout_class
                features[accession] = row
    print(f"proteins with an A/B/C call: {len(reference):,}", flush=True)

    store = StringStore()
    payload = json.loads((OUT / "string_store.json").read_text())
    for accession, rows in payload["partners"].items():
        store.partners[accession] = [Partner.from_json_dict(r) for r in rows]
    store.failures.update(payload.get("failures", {}))
    print(
        f"STRING: {len(store):,} with an entry, {store.n_absent:,} absent, {store.n_failed:,} failed",
        flush=True,
    )

    curated = store.names_by_accession(CURATED_CHANNELS)
    context = store.names_by_accession(CONTEXT_CHANNELS)
    print(f"  curated-evidence partners: {len(curated):,}   genomic-context: {len(context):,}", flush=True)

    for label, source in (("curated", curated), ("context", context)):
        rates = tie_rate(source, lambda g: classify_partner(g, PARALOGUE))
        print(
            f"  ties in {label} paralogue readout: {rates['n_tied']:,} of {rates['n_labelled']:,} "
            f"({rates['tie_fraction']:.1%}) dropped",
            flush=True,
        )

    # Whichever channel is chosen for discovery, the other validates - so the two runs are
    # genuine replications on independent evidence rather than one confirming itself.
    channel = os.environ.get("STRING_DISCOVERY_CHANNEL", "curated")
    if channel not in {"curated", "context"}:
        message = f"STRING_DISCOVERY_CHANNEL must be 'curated' or 'context', got {channel!r}"
        raise ValueError(message)
    ranking_source, other_source = (curated, context) if channel == "curated" else (context, curated)
    other_name = "context" if channel == "curated" else "curated"
    print(f"\ndiscovery channel: {channel}   validating channel: {other_name}", flush=True)

    discovery = dominant_label(ranking_source, lambda g: classify_partner(g, PARALOGUE))
    held_out = {
        f"{other_name}_paralogue_partner": dominant_label(other_source, lambda g: classify_partner(g, PARALOGUE)),
        "dominant_non_hsp70_system": dominant_label(ranking_source, classify_non_hsp70),
        "interactome_scope": interactome_scope(ranking_source, is_hsp70),
    }
    print(
        f"\ndiscovery: {len(discovery):,} proteins; held out: { {k: len(v) for k, v in held_out.items()} }",
        flush=True,
    )

    candidates = candidate_partitions(features)
    print(f"candidate partitions: {len(candidates)}", flush=True)

    reports = {}
    for criterion in (CRITERION_V, CRITERION_INFORMATION):
        report = search_partitions(
            candidates,
            reference,
            discovery,
            held_out,
            discovery_name=f"string_{channel}_paralogue",
            n_permutations=500,
            criterion=criterion,
        )
        report["relationship_to_reference"] = {
            name: partition_relationship(part, reference) for name, part in candidates.items()
        }
        reports[criterion] = report
    (OUT / f"partition_search_string_{channel}.json").write_text(json.dumps(reports, indent=2) + "\n")

    for criterion, report in reports.items():
        print(f"\n{'=' * 92}\n{criterion.upper()}\n{'=' * 92}")
        print("%-40s %6s %8s %8s %8s %8s %6s" % ("candidate", "grps", "n", "V(cand)", "V(ABC)", "bits", "beats"))
        for d in report["discovery"][:12]:
            beats = d["beats_reference_on_information"] if criterion == CRITERION_INFORMATION else d["beats_reference"]
            print(
                "  %-38s %6d %8d %8.4f %8.4f %8.4f %6s"
                % (
                    d["candidate"][:37],
                    d["n_groups"],
                    d["n_proteins"],
                    d["candidate_cramers_v"],
                    d["reference_cramers_v"],
                    d.get("candidate_bits", 0.0),
                    "YES" if (beats and d["beats_null"]) else "",
                )
            )
        print(f"\n  confirmed on held-out readouts: {report['confirmed_candidates']}")

    print(f"\n{'=' * 92}\nRELATIONSHIP TO A/B/C\n{'=' * 92}")
    confirmed = set(reports[CRITERION_INFORMATION]["confirmed_candidates"])
    for name in sorted(confirmed):
        kind = reports[CRITERION_INFORMATION]["relationship_to_reference"].get(name, "?")
        print(f"  {name:40s} {kind:12s} {describe_relationship(kind)}")

    print(f"\nwrote {OUT / f'partition_search_string_{channel}.json'}")


if __name__ == "__main__":
    main()
