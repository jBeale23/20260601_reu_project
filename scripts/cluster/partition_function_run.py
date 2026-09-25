"""Do the winning partitions predict function better than A/B/C does?

Partner identity is one readout family. A grouping that beats A/B/C there has shown it
tracks who a protein works with; it has not shown it tracks what the protein does. Function
is a different kind of evidence and the partitions were never selected against it.

Terms come from the phylogenetic tier or better - experiment-grounded, propagated through
trees rather than through our own sequence profiles. The electronic tier is excluded
throughout: its commonest route is InterPro2GO, so a domain signature implies the term, and
testing a domain-architecture partition against it would assume the answer.

Each partition is scored by the information it carries about a term, above a null built by
shuffling that partition's own labels. Shuffling preserves group sizes, so a partition cannot
win for being finer - the control that mattered on the partner readout matters here too.
"""

import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from data_fetching.fetch_go_evidence import TIER_PHYLOGENETIC, EvidenceRecord, EvidenceStore
from validation.modality_information import mutual_information
from validation.partition_search import MIN_RELATIVE_MARGIN, candidate_partitions

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

# The partitions confirmed on the STRING partner readout.
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

MIN_TERM = 60
N_NULL = 60
MAX_TERMS = 30


def information_above_null(labels, target, rng):
    """Information a labelling carries about a target, above its own shape-matched null."""
    observed = mutual_information(labels, target)
    shuffled = list(labels)
    draws = []
    for _ in range(N_NULL):
        rng.shuffle(shuffled)
        draws.append(mutual_information(shuffled, target))
    return max(0.0, observed - sum(draws) / len(draws))


def main():
    features, reference = {}, {}
    with (WK / "full_layout_v6" / "domain_layout_features.csv").open() as fh:
        for row in csv.DictReader(fh):
            accession = (row.get("accession") or "").strip()
            layout_class = (row.get("layout_predicted_class") or "").strip()
            if accession:
                features[accession] = row
                if layout_class in {"A", "B", "C"}:
                    reference[accession] = layout_class

    evidence = EvidenceStore()
    payload = json.loads((OUT / "go_evidence_store.json").read_text())
    for accession, record in payload["records"].items():
        evidence.records[accession] = EvidenceRecord.from_json_dict(record)
    terms_by = evidence.terms_by_accession(TIER_PHYLOGENETIC, cumulative=True)
    print(f"proteins with phylogenetic-or-better terms: {len(terms_by):,}", flush=True)

    candidates = candidate_partitions(features)
    schemes = {"A/B/C": reference} | {n: candidates[n] for n in WINNERS if n in candidates}

    shared = sorted(set(terms_by) & set(reference))
    counts = Counter(t for a in shared for t in set(terms_by[a]))
    chosen = [t for t, c in counts.most_common(MAX_TERMS) if c >= MIN_TERM]
    print(f"proteins scored: {len(shared):,}   terms tested: {len(chosen)}", flush=True)

    rng = random.Random(0)  # noqa: S311 - a permutation null, not a secret
    per_scheme = {}
    for name, partition in schemes.items():
        usable = [a for a in shared if a in partition]
        labels = [partition[a] for a in usable]
        bits = {}
        for term in chosen:
            target = [1 if term in set(terms_by[a]) else 0 for a in usable]
            if 0 < sum(target) < len(target):
                bits[term] = information_above_null(labels, target, rng)
        per_scheme[name] = {"n": len(usable), "n_groups": len(set(labels)), "bits": bits}
        print(f"  {name[:44]:46s} n={len(usable):6,}  mean bits={sum(bits.values()) / max(1, len(bits)):.4f}")

    baseline = per_scheme["A/B/C"]["bits"]
    print(f"\n  {'scheme':46s} {'terms won':>10s} {'mean margin':>13s}  beats A/B/C")
    summary = {}
    for name, data in per_scheme.items():
        if name == "A/B/C":
            continue
        shared_terms = [t for t in data["bits"] if t in baseline and baseline[t] > 0]
        margins = [(data["bits"][t] - baseline[t]) / baseline[t] for t in shared_terms]
        won = sum(1 for m in margins if m >= MIN_RELATIVE_MARGIN)
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        verdict = "YES" if won > len(margins) / 2 else ""
        summary[name] = {
            "terms_compared": len(margins),
            "terms_beating_reference": won,
            "mean_relative_margin": round(mean_margin, 4),
            "beats_reference_on_most_terms": bool(verdict),
        }
        print(f"  {name[:44]:46s} {won:4d}/{len(margins):<5d} {mean_margin:+12.1%}  {verdict}")

    (OUT / "partition_function.json").write_text(
        json.dumps({"per_scheme": per_scheme, "summary": summary}, indent=2) + "\n"
    )
    print(f"\nwrote {OUT / 'partition_function.json'}")


if __name__ == "__main__":
    main()
