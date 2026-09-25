"""Do grammar and structure say the same thing about function, or different things?

Runs the interaction-information split over the commonest functional terms, once per
evidence tier. The tiers are never pooled: they are different kinds of evidence and the
difference decides what a result means.

``experimental``
    Somebody did an experiment on this protein. The only tier that cannot be circular with a
    claim about domain architecture. QuickGO supplies 344 such proteins where UniProt's
    per-entry route found 198.
``phylogenetic``
    Propagated from an experimentally annotated relative through a phylogenetic tree. Nearly
    4,000 proteins, and grounded in experiment at one remove. Trees come from whole-sequence
    phylogeny rather than domain signatures.
``homology_transfer``
    Our own HMM transfer. Kept last because it used sequence profiles, which gives the
    sequence-derived grammar an inside track on the labels.

The two profiles are chosen so the question is not answered by construction:

- **Grammar** is ``idr_fraction`` and ``n_structured_domains`` - how much of the chain is
  disordered and how many folded blocks it carries. Both come from the sequence and the
  InterPro boundaries; neither needs a structure.
- **Structure** is ``relative_contact_order`` and ``fraction_buried`` - fold topology and
  packing. Neither is recoverable from composition.

Deliberately *not* paired: ``idr_fraction`` against ``fraction_disordered_plddt``. Those two
measure the same property by two routes, so pairing them would guarantee a redundant answer
and would be measuring the agreement of two disorder predictors rather than anything about
function.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_go_evidence import (
    TIER_EXPERIMENTAL,
    TIER_PHYLOGENETIC,
    EvidenceRecord,
    EvidenceStore,
)
from validation.modality_information import Modality, ReportSettings, modality_report

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
TRANSFER = Path("/home/jbeale3/transfer_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

LAYOUT = WK / "full_layout_v6" / "domain_layout_features.csv"
STRUCTURE = WK / "structure_features_surface.csv"

GRAMMAR_PROFILE = ["idr_fraction", "n_structured_domains"]
STRUCTURE_PROFILE = ["relative_contact_order", "fraction_buried"]


def numeric_rows(path: Path, wanted: list[str]) -> dict[str, dict[str, float]]:
    """Read only the columns needed, coercing to float and skipping unusable rows."""
    out: dict[str, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            values = {}
            for name in wanted:
                try:
                    values[name] = float(row[name])
                except (KeyError, TypeError, ValueError):
                    break
            else:
                out[accession] = values
    return out


def main():
    grammar_features = numeric_rows(LAYOUT, GRAMMAR_PROFILE)
    structure_features = numeric_rows(STRUCTURE, STRUCTURE_PROFILE)
    print(f"grammar features: {len(grammar_features)}   structure features: {len(structure_features)}", flush=True)

    evidence = EvidenceStore()
    payload = json.loads((OUT / "go_evidence_store.json").read_text())
    for accession, record in payload["records"].items():
        evidence.records[accession] = EvidenceRecord.from_json_dict(record)
    print(f"GO evidence store: {len(evidence)} records, tiers {evidence.tier_counts()}", flush=True)

    experimental = evidence.terms_by_accession(TIER_EXPERIMENTAL)
    # Cumulative: phylogenetic terms plus the experimental ones, since a protein with both
    # should be scored on everything it has rather than on the weaker half.
    phylogenetic = evidence.terms_by_accession(TIER_PHYLOGENETIC, cumulative=True)

    transferred_raw = json.loads((TRANSFER / "function_transfer.json").read_text())
    transferred = {
        accession: tuple(item.get("terms") or ()) for accession, item in transferred_raw.items() if item.get("terms")
    }
    print(
        f"experimental: {len(experimental)}   phylogenetic(+exp): {len(phylogenetic)}   "
        f"homology transfer: {len(transferred)}",
        flush=True,
    )

    grammar = Modality(grammar_features, GRAMMAR_PROFILE)
    structure = Modality(structure_features, STRUCTURE_PROFILE)

    reports = {}
    tiers = (
        ("experimental", experimental),
        ("phylogenetic", phylogenetic),
        ("homology_transfer", transferred),
    )
    for name, terms in tiers:
        print(f"\n{'=' * 78}\n{name.upper()} TERMS\n{'=' * 78}", flush=True)
        report = modality_report(
            terms,
            grammar,
            structure,
            settings=ReportSettings(n_permutations=200, provenance=name),
        )
        reports[name] = report
        print(
            f"  proteins: {report['n_proteins']}   terms tested: {report.get('n_terms_tested', 0)}   "
            f"grammar states: {report.get('n_grammar_states', 0)}   "
            f"structure states: {report.get('n_structure_states', 0)}",
            flush=True,
        )
        if not report.get("results"):
            print(f"  {report['interpretation']}", flush=True)
            continue
        print(f"  summary: {report['summary']}", flush=True)
        print("  %-46s %8s %8s %8s %10s %s" % ("term", "I(G;Y)", "I(S;Y)", "I(GS;Y)", "interact", "relationship"))
        rows = sorted(report["results"], key=lambda r: -r["joint_bits"])
        for r in rows[:20]:
            print(
                "    %-44s %8.4f %8.4f %8.4f %10.4f %s"
                % (
                    r["term"][:44],
                    r["grammar_bits"],
                    r["structure_bits"],
                    r["joint_bits"],
                    r["interaction_bits"],
                    r["relationship"] if r["significant"] else "not significant",
                )
            )
        print(f"\n  {report['interpretation']}", flush=True)

    (OUT / "modality_information.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"\nwrote {OUT / 'modality_information.json'}")


if __name__ == "__main__":
    main()
