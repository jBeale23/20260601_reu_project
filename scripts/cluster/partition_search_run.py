"""Search for a grouping that beats A/B/C, then describe what it is biologically."""

# Vendored from a scratch-only copy on Rockfish. It had been living in ~/.inspect, which is
# how the previous driver nearly went missing; every result it produces is now reproducible
# from the repository.
#
# Outputs honour ANALYSIS_OUT_DIR because the scr4_sfried3 group quota is shared with the
# rest of the lab and fills without warning - when it does, writes fail while reads keep
# working, so a run computes for hours and then has nowhere to put the answer.
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
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
LAYOUT = WK / "full_layout_v6" / "domain_layout_features.csv"


def main():
    features, reference = {}, {}
    with LAYOUT.open() as fh:
        for row in csv.DictReader(fh):
            a = (row.get("accession") or "").strip()
            c = (row.get("layout_predicted_class") or "").strip()
            if a and c in {"A", "B", "C"}:
                reference[a] = c
                features[a] = row
    print(f"proteins with an A/B/C call: {len(reference)}", flush=True)

    inter = json.loads((WK / "biogrid_interactions_by_accession.json").read_text())["interactions"]

    # Readouts come from validation.partner_readouts so that this script and the
    # sufficiency run reduce the interaction map identically; defined separately they drift,
    # and two Cramer's V values stop being about the same thing without either run failing.

    # A dominant-partner readout is only as good as its unambiguity. Ties are dropped rather
    # than broken arbitrarily, so say how many proteins that cost.
    for label, mapper in (
        ("hsp70_paralogue_partner", lambda g: classify_partner(g, PARALOGUE)),
        ("dominant_non_hsp70_system", classify_non_hsp70),
    ):
        rates = tie_rate(inter, mapper)
        print(
            f"  ties in {label}: {rates['n_tied']} of {rates['n_labelled']} "
            f"({rates['tie_fraction']:.1%}) dropped as having no dominant partner",
            flush=True,
        )

    discovery = dominant_label(inter, lambda g: classify_partner(g, PARALOGUE))
    held_out = {
        "dominant_non_hsp70_system": dominant_label(inter, classify_non_hsp70),
        "interactome_scope": interactome_scope(inter, is_hsp70),
    }
    print(f"discovery: {len(discovery)} proteins; held out: { {k: len(v) for k, v in held_out.items()} }", flush=True)

    candidates = candidate_partitions(features)
    print(f"candidate partitions: {sorted(candidates)}", flush=True)

    # Both criteria, side by side, rather than a choice between them. Cramer's V asks how
    # close an association is to the maximum its table shape allows and so penalises a finer
    # partition; mutual information asks only how much the partition tells you about the
    # readout. Combinations are always finer than their parents, so the two can disagree -
    # and running both means the disagreement is data rather than a decision made after
    # seeing which answer was preferable.
    reports = {}
    for criterion in (CRITERION_V, CRITERION_INFORMATION):
        report = search_partitions(
            candidates,
            reference,
            discovery,
            held_out,
            discovery_name="hsp70_paralogue_partner",
            n_permutations=500,
            criterion=criterion,
        )
        report["relationship_to_reference"] = {
            name: partition_relationship(part, reference) for name, part in candidates.items()
        }
        reports[criterion] = report
    (OUT / "partition_search.json").write_text(json.dumps(reports, indent=2) + "\n")

    report = reports[CRITERION_V]

    print()
    print("=" * 86)
    print("DISCOVERY  (ranked on Hsp70 paralogue partner)")
    print("=" * 86)
    # Two verdicts, because they answer different questions and can disagree. Cramer's V
    # asks how close the association is to the maximum its table shape allows, and so
    # penalises a finer partition even when the extra groups carry real signal. Mutual
    # information asks only how much the partition tells you about the readout - the right
    # question for a search over feature combinations, which are always finer than their
    # parents. A combination that wins on bits but loses on V is gaining information and
    # paying for it in granularity.
    print(
        "%-34s %6s %7s %8s %8s %8s %8s %9s %6s"
        % ("candidate", "grps", "n", "V(cand)", "V(ABC)", "bits", "bitsABC", "p_adj", "beats")
    )
    for d in report["discovery"]:
        verdict = "YES" if (d["beats_reference"] and d["beats_null"]) else ""
        if d.get("beats_reference_on_information") and not d["beats_reference"]:
            verdict = "bits"
        print(
            "  %-32s %6d %7d %8.4f %8.4f %8.4f %8.4f %9.4f %6s"
            % (
                d["candidate"],
                d["n_groups"],
                d["n_proteins"],
                d["candidate_cramers_v"],
                d["reference_cramers_v"],
                d.get("candidate_bits", 0.0),
                d.get("reference_bits", 0.0),
                d["p_adjusted"],
                verdict,
            )
        )

    on_bits = [d["candidate"] for d in report["discovery"] if d.get("beats_reference_on_information")]
    print(f"\n  candidates carrying more information about the readout than A/B/C: {len(on_bits)}")
    if on_bits:
        print(f"    {', '.join(on_bits[:8])}")
    print()
    print("=" * 86)
    print("VALIDATION  (readouts held back from ranking)")
    print("=" * 86)
    for v in report["validation"]:
        print(
            "  %-32s %-28s V=%.4f vs ABC %.4f p_adj=%.4f %s"
            % (
                v["candidate"],
                v["readout"],
                v["candidate_cramers_v"],
                v["reference_cramers_v"],
                v["p_adjusted"],
                "CONFIRMED" if (v["beats_reference"] and v["beats_null"]) else "",
            )
        )
    print()
    print(f"CONFIRMED CANDIDATES: {report['confirmed_candidates']}")

    # A candidate that scores better has not necessarily found a new division. It may be the
    # reference with two classes merged or one split, and those mean opposite things - so
    # say which, rather than leaving a coarsening to read as a discovery.
    print()
    print("=" * 86)
    print("RELATIONSHIP TO A/B/C")
    print("=" * 86)
    for name in sorted(candidates):
        kind = partition_relationship(candidates[name], reference)
        mark = "  <-- confirmed" if name in report["confirmed_candidates"] else ""
        print(f"  {name:34s} {kind:12s} {describe_relationship(kind)}{mark}")

    print()
    print("=" * 86)
    print("THE TWO CRITERIA, SIDE BY SIDE")
    print("=" * 86)
    for criterion, other in reports.items():
        print(
            f"  {criterion:12s}: {other['n_beating_reference_on_discovery']:2d} beat A/B/C at discovery, "
            f"{other['n_confirmed_on_held_out_readout']:2d} confirmed on a held-out readout "
            f"-> {other['confirmed_candidates']}"
        )
    v_disc = {d["candidate"] for d in reports[CRITERION_V]["discovery"] if d["beats_reference"]}
    bits_disc = {
        d["candidate"] for d in reports[CRITERION_INFORMATION]["discovery"] if d["beats_reference_on_information"]
    }
    only_bits = sorted(bits_disc - v_disc)
    print(f"\n  beat A/B/C on information but not on V ({len(only_bits)}):")
    for name in only_bits:
        print(f"    {name}")
    print("  These are the candidates V rejects for being finer while they carry more signal.")

    for name in report["confirmed_candidates"]:
        part = candidates[name]
        print()
        print("=" * 86)
        print(f"BIOLOGY OF: {name}")
        print("=" * 86)
        groups = defaultdict(list)
        for a, g in part.items():
            groups[g].append(a)
        for g, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            abc = Counter(reference[a] for a in members if a in reference)
            partners = Counter(discovery[a] for a in members if a in discovery)
            systems = Counter(
                held_out["dominant_non_hsp70_system"].get(a)
                for a in members
                if a in held_out["dominant_non_hsp70_system"]
            )
            print(f"  {g}: {len(members):,} proteins")
            print(f"      A/B/C mix     : {dict(abc.most_common(3))}")
            print(f"      top partners  : {dict(partners.most_common(4))}")
            print(f"      top systems   : {dict(systems.most_common(4))}")


if __name__ == "__main__":
    main()
