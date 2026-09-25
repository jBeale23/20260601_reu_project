"""Is A/B/C sufficient, and is the finer split better? Tested against independent readouts."""

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
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from validation.class_sufficiency import sufficiency_report
from validation.pairing import PARALOGUE, classify_partner, is_hsp70
from validation.partner_readouts import dominant_label, interactome_scope, tie_rate
from validation.partner_systems import classify_non_hsp70

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
LAYOUT = WK / "full_layout_v6" / "domain_layout_features.csv"


def main():
    coarse, fine = {}, {}
    with LAYOUT.open() as fh:
        for row in csv.DictReader(fh):
            a = (row.get("accession") or "").strip()
            c = (row.get("layout_predicted_class") or "").strip()
            s = (row.get("layout_predicted_subclass") or "").strip()
            if a and c in {"A", "B", "C"} and s:
                coarse[a], fine[a] = c, s
    print(
        f"classified: {len(coarse)}  coarse={sorted(set(coarse.values()))}  fine={len(set(fine.values()))} classes",
        flush=True,
    )

    inter = json.loads((WK / "biogrid_interactions_by_accession.json").read_text())["interactions"]

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

    # Shared with the partition search: see validation.partner_readouts.
    readouts = {
        "hsp70_paralogue_partner": dominant_label(inter, lambda g: classify_partner(g, PARALOGUE)),
        "dominant_non_hsp70_system": dominant_label(inter, classify_non_hsp70),
        "interactome_scope": interactome_scope(inter, is_hsp70),
    }
    for name, r in readouts.items():
        print(f"  readout {name}: {len(r)} proteins, {len(set(r.values()))} categories", flush=True)

    report = sufficiency_report(coarse, fine, readouts, n_permutations=1000)
    (OUT / "class_sufficiency.json").write_text(json.dumps(report, indent=2) + "\n")

    print()
    print("=" * 78)
    print("IS A/B/C SUFFICIENT?  (independent readouts, shape-matched null)")
    print("=" * 78)
    print("%-30s %6s %8s %8s %9s %9s %s" % ("readout", "n", "V(ABC)", "V(fine)", "nullMean", "p", "beats?"))
    for r in report["results"]:
        print(
            "  %-28s %6d %8.4f %8.4f %9.4f %9.4f %s"
            % (
                r["readout"],
                r["n_proteins"],
                r["coarse_cramers_v"],
                r["fine_cramers_v"],
                r["null_mean_cramers_v"],
                r["p_value"],
                r["beats_random_refinement"],
            )
        )
    print()
    print(
        f"readouts where the split beats a random refinement: "
        f"{report['n_readouts_where_split_beats_random']} of {len(report['results'])}"
    )

    # Beating a random refinement is a weak bar: it only says the split is not arbitrary.
    # The question the project actually asks is whether the finer classes beat A/B/C, and
    # on that the two comparisons can disagree - a split can be non-arbitrary and still
    # worse than the scheme it refines.
    better = [r["readout"] for r in report["results"] if r["fine_cramers_v"] > r["coarse_cramers_v"]]
    worse = [r["readout"] for r in report["results"] if r["fine_cramers_v"] < r["coarse_cramers_v"]]
    print(f"readouts where the finer split beats A/B/C itself: {len(better)} of {len(report['results'])}")
    if better:
        print(f"  better: {', '.join(better)}")
    if worse:
        print(f"  worse : {', '.join(worse)}")
    if worse and not better:
        print("  -> the finer split is not an improvement on A/B/C for any readout tested.")


if __name__ == "__main__":
    main()
