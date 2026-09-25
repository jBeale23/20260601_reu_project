"""How much of each class scheme can be recovered from features, block by block.

The partition search asks whether a grouping predicts something external. This asks the
converse, and for the project's stated goal it is the more important question: given only
features you could measure on a protein from an organism nobody has studied, how much of the
class assignment can you recover?

A scheme that carves biology beautifully but cannot be predicted from sequence is useless for
the thing this project exists to do. A/B/C sets the bar - it is nearly a deterministic
function of two domain flags, so it should be almost perfectly recoverable, and any richer
scheme has to be judged against that ceiling rather than against zero.

Reported as the uncertainty coefficient, I(features; class) / H(class): the share of the
class label's entropy the features account for. Dividing by the label's own entropy is what
makes a three-group scheme and a twelve-group one comparable - raw information would simply
reward whichever scheme is finer.

Feature blocks are kept apart because they cost different things to obtain:

``grammar``      sequence and InterPro boundaries only - available for any protein
``structure``    needs a folded model, available for 74% of this set
``electrostatic`` needs a model *and* a Poisson-Boltzmann solve

One correction is load-bearing. Several candidate partitions are *built from* grammar
features - ``idr_tertile+n_structured_domains`` is a function of two of them - so asking how
well grammar recovers that partition is asking whether a variable predicts itself. The first
run of this analysis reported 0.89 for such a scheme against A/B/C's 0.32 and the number
meant nothing. Any feature a partition was built from is therefore dropped from the block
before scoring it, and the columns that were dropped are reported alongside the score so a
reader can see what each number was actually computed on.
"""

import csv
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/jbeale3/repositories/reu_domain_layout_v2")
from validation.electrostatics import load_energies
from validation.modality_information import Modality, mutual_information
from validation.partition_search import BINARY_COLUMNS, WIDE_NUMERIC_COLUMNS, candidate_partitions

WK = Path("/home/jbeale3/scr4_sfried3/domain_layout_run")
APBS = Path("/home/jbeale3/apbs_jdp/outputs")
OUT = Path(os.environ.get("ANALYSIS_OUT_DIR", str(WK)))
OUT.mkdir(parents=True, exist_ok=True)

WINNERS = [
    "top_architectures",
    "has_dnaj_c+n_structured_domains",
    "has_zinc_finger_like+idr_tertile",
    "idr_tertile+n_structured_domains",
    "has_dnaj_c+idr_tertile",
    "has_gf_rich_region+n_structured_domains",
]

# The minimal grammar: what the layout stage reads off a sequence before either layer runs.
GRAMMAR = ["idr_fraction", "n_structured_domains", "mean_disorder"]

# The full hierarchical grammar, adding what the dual-layer routing produced: how many
# regions went to alignment against how many went to the alignment-free comparison, and how
# similar the best alignment-free match was. Those three are the actual output of the two
# layers, so this block is the fair test of whether the hierarchical model buys anything
# over reading disorder and domain count off the sequence.
GRAMMAR_ROUTING = [
    "idr_fraction",
    "n_structured_domains",
    "mean_disorder",
    "n_msa_regions",
    "n_shark_regions",
]

# Adds similarity to the nearest alignment-free reference. Reported separately from
# GRAMMAR_ROUTING because that reference set is built from class exemplars: the class of the
# best match is excluded outright as a classifier output, but the *similarity* to it is still
# reference-set dependent and could carry class identity indirectly. Comparing the two blocks
# is what says whether the routing counts are doing the work or the reference match is.
GRAMMAR_FULL = [*GRAMMAR_ROUTING, "shark_best_similarity"]
STRUCTURE = ["relative_contact_order", "fraction_buried", "mean_plddt", "compactness"]

# Permutations for the bias floor. Fewer than the search uses: this estimates a floor, not a
# p-value, and each draw is a contingency table over every protein.
N_NULL = 40


def entropy_bits(labels):
    """Shannon entropy of a labelling, in bits."""
    counts = Counter(labels)
    total = sum(counts.values())
    out = 0.0
    for count in counts.values():
        if count:
            p = count / total
            out -= p * math.log2(p)
    return out


def numeric_rows(path: Path, wanted: list[str]) -> dict[str, dict[str, float]]:
    """Read only the wanted columns, skipping rows where any is unusable."""
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


def inputs_of(scheme_name: str) -> set[str]:
    """Which feature columns a candidate partition was built from.

    Read off the candidate's name, which is how the search composes them: a partition called
    ``idr_tertile+n_structured_domains`` is a function of exactly those two columns. Scoring
    a block that still contains them measures the definition, not a prediction.
    """
    used: set[str] = set()
    for part in scheme_name.split("+"):
        stem = part.removesuffix("_tertile")
        for column, label in WIDE_NUMERIC_COLUMNS:
            if stem in {column, label}:
                used.add(column)
        if part in BINARY_COLUMNS or part in {"n_structured_domains", "j_domain_position"}:
            used.add(part)
    return used


def recoverable(block: Modality, partition, accessions, *, exclude: set[str]) -> float:
    """Share of a partition's entropy the feature block accounts for, above its null.

    Columns in ``exclude`` are dropped first: they are the partition's own inputs, and a
    block that still holds them scores the definition rather than a prediction.
    """
    names = [n for n in block.names if n not in exclude]
    if not names:
        return float("nan")
    block = Modality(block.features, names)
    shared = [a for a in accessions if a in block.features and a in partition]
    if len(shared) < 200:
        return float("nan")
    states = block.states(shared)
    labels = [partition[a] for a in shared]
    observed = mutual_information(states, labels)

    rng = random.Random(0)  # noqa: S311 - a permutation null, not a secret
    shuffled = list(labels)
    draws = []
    for _ in range(N_NULL):
        rng.shuffle(shuffled)
        draws.append(mutual_information(states, shuffled))
    excess = max(0.0, observed - sum(draws) / len(draws))
    h = entropy_bits(labels)
    return excess / h if h > 0 else 0.0


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

    layout_csv = WK / "full_layout_v6" / "domain_layout_features.csv"
    grammar = Modality(numeric_rows(layout_csv, GRAMMAR), GRAMMAR)
    grammar_routing = Modality(numeric_rows(layout_csv, GRAMMAR_ROUTING), GRAMMAR_ROUTING)
    grammar_full = Modality(numeric_rows(layout_csv, GRAMMAR_FULL), GRAMMAR_FULL)
    structure = Modality(numeric_rows(WK / "structure_features_surface.csv", STRUCTURE), STRUCTURE)
    energies = load_energies(sorted(APBS.glob("jdp_energies_*.tsv")))
    lengths = {a: float(features[a]["protein_length"]) for a in energies if a in features}
    electro = Modality(
        {a: {"solvation_per_residue": energies[a].solvation / lengths[a]} for a in lengths if lengths[a] > 0},
        ["solvation_per_residue"],
    )
    print(
        f"grammar {len(grammar.features):,}   routing {len(grammar_routing.features):,}   "
        f"full {len(grammar_full.features):,}   "
        f"structure {len(structure.features):,}   electrostatic {len(electro.features):,}",
        flush=True,
    )

    candidates = candidate_partitions(features, numeric_columns=WIDE_NUMERIC_COLUMNS)
    print(f"candidate partitions built: {len(candidates)}", flush=True)
    schemes = {"A/B/C": reference} | {name: candidates[name] for name in WINNERS if name in candidates}

    blocks = {
        "grammar": grammar,
        "grammar_routing": grammar_routing,
        "grammar_full": grammar_full,
        "structure": structure,
        "electrostatic": electro,
    }
    report = {}
    print(f"\n  {'scheme':42s} {'groups':>7s} " + " ".join(f"{b:>14s}" for b in blocks) + "   own inputs dropped")
    for name, partition in schemes.items():
        exclude = inputs_of(name)
        row = {}
        for block_name, block in blocks.items():
            row[block_name] = recoverable(block, partition, sorted(block.features), exclude=exclude)
        report[name] = {
            "n_groups": len(set(partition.values())),
            "recoverable": row,
            "excluded_own_inputs": sorted(exclude),
        }
        cells = " ".join(f"{row[b]:14.3f}" if row[b] == row[b] else f"{'n/a':>14s}" for b in blocks)
        dropped = ", ".join(sorted(exclude)) or "none"
        print(f"  {name[:41]:42s} {len(set(partition.values())):7d} {cells}   {dropped}")

    (OUT / "class_predictability.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nShare of each scheme's entropy recovered from each feature block, above its own null.")
    print(f"wrote {OUT / 'class_predictability.json'}")


if __name__ == "__main__":
    main()
