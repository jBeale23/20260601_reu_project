"""Does Poisson-Boltzmann solvation predict the new partitions, disorder, and function?

The first electrostatics run asked whether solvation density separates the A/B/C subclasses.
It does, for c_j_domain_only, and survives a disorder control. This asks the three follow-on
questions:

1. Do the six partitions that beat A/B/C on partner information also separate on solvation?
   A grouping found from domain content and confirmed on interaction data has no claim on an
   electrostatic property; if it separates there too, three independent measurements agree.
2. Does solvation track disorder across the whole set, and does it survive that control?
3. Does solvation carry information about function, and is that information the same
   information the grammar carries or different?

Every comparison is on solvation *per residue*. Raw energy tracks chain length at rho -0.91,
so a raw comparison across groups of different sizes measures length and nothing else.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_go_evidence import TIER_PHYLOGENETIC, EvidenceRecord, EvidenceStore
from validation.electrostatics import (
    compare_groups,
    effect_label,
    load_energies,
    stratified_by_covariate,
)
from validation.modality_information import Modality, ReportSettings, modality_report
from validation.partition_search import candidate_partitions

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
APBS = Path("/home/jbeale3/apbs_jdp/outputs")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

# The partitions that beat A/B/C on partner information and confirmed on held-out readouts.
WINNERS = [
    "top_architectures",
    "has_dnaj_c+n_structured_domains",
    "has_zinc_finger_like+idr_tertile",
    "idr_tertile+n_structured_domains",
    "has_dnaj_c+idr_tertile",
    "has_gf_rich_region+n_structured_domains",
]


def main():
    energies = load_energies(sorted(APBS.glob("jdp_energies_*.tsv")))
    print(f"solvation energies: {len(energies):,}", flush=True)

    features, lengths, disorder = {}, {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            features[accession] = row
            if accession not in energies:
                continue
            try:
                lengths[accession] = int(row["protein_length"])
                disorder[accession] = float(row["mean_disorder"])
            except (KeyError, TypeError, ValueError):
                continue

    density = {a: energies[a].solvation / lengths[a] for a in lengths if lengths[a] > 0}
    print(f"with length and disorder: {len(density):,}", flush=True)

    report = {"n_with_solvation_density": len(density), "partitions": {}}

    candidates = candidate_partitions(features)
    print(f"\n{'=' * 92}\nDOES SOLVATION SEPARATE THE PARTITIONS THAT BEAT A/B/C ON PARTNERS?\n{'=' * 92}")
    for name in WINNERS:
        partition = candidates.get(name)
        if not partition:
            continue
        groups = {a: partition[a] for a in density if a in partition}
        # compare_groups keys off the group map, so passing the full density table is safe.
        comparisons = compare_groups(density, groups, "solvation_per_residue")
        significant = [c for c in comparisons if c.significant]
        # The disorder control, since solvation density and disorder are correlated and the
        # partitions are partly built from disorder tertiles.
        controlled = stratified_by_covariate(density, groups, disorder, "solvation_per_residue")
        report["partitions"][name] = {
            "n_groups_tested": len(comparisons),
            "n_significant": len(significant),
            "consistent_across_disorder": controlled["consistent_groups"],
            "comparisons": [c.to_json_dict() for c in comparisons],
        }
        print(
            f"\n  {name}: {len(significant)} of {len(comparisons)} groups separate on solvation; "
            f"{len(controlled['consistent_groups'])} hold at every disorder level"
        )
        for c in sorted(significant, key=lambda x: x.delta)[:4]:
            mark = "  <-- survives disorder control" if c.group in controlled["consistent_groups"] else ""
            print(
                f"      {c.group[:34]:36s} n={c.n_in:5d} median={c.median_in:8.2f} "
                f"delta={c.delta:+.3f} {effect_label(c.delta)}{mark}"
            )

    # Does solvation carry information about function, and is it the grammar's information?
    print(f"\n{'=' * 92}\nDOES SOLVATION CARRY FUNCTION INFORMATION THE GRAMMAR DOES NOT?\n{'=' * 92}")
    evidence = EvidenceStore()
    payload = json.loads((OUT / "go_evidence_store.json").read_text())
    for accession, record in payload["records"].items():
        evidence.records[accession] = EvidenceRecord.from_json_dict(record)
    terms = evidence.terms_by_accession(TIER_PHYLOGENETIC, cumulative=True)

    grammar = Modality(
        {a: {"idr_fraction": float(features[a]["idr_fraction"])} for a in density if a in features},
        ["idr_fraction"],
    )
    electro = Modality({a: {"solvation_per_residue": density[a]} for a in density}, ["solvation_per_residue"])
    functional = modality_report(
        terms, grammar, electro, settings=ReportSettings(n_permutations=200, provenance="phylogenetic")
    )
    report["solvation_vs_function"] = functional
    print(f"  proteins: {functional['n_proteins']}   terms tested: {functional.get('n_terms_tested', 0)}")
    print(f"  summary: {functional.get('summary')}")
    print(f"  {functional['interpretation']}")

    (OUT / "electrostatics_extended.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT / 'electrostatics_extended.json'}")


if __name__ == "__main__":
    main()
