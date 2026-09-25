"""Where sequence, structure, and function stop agreeing.

Sequence, fold, and function usually change together, and when they do there is nothing to
learn: divergent sequence, divergent structure, divergent function is the ordinary case.
The informative proteins are the ones where the three come apart, and the two ways they can
come apart mean opposite things.

**Sequence diverged, structure conserved.** The fold tolerated substitution. These pairs
mark the positions that were free to change - the protein's *permissive* sequence. Finding
many such pairs in one subpopulation and few in another says the two are under different
structural constraint.

**Sequence conserved, structure diverged.** A small number of substitutions moved the fold.
Those substitutions are candidates for the ones that *matter*, because the same sequence
change elsewhere in the protein did nothing. This is the rarer and more interesting case,
and it is where a change in function is most likely to be found.

Adding function to the pair makes the question sharper still: among pairs that kept their
structure, which ones also kept their function, and what distinguishes those from the pairs
that changed function anyway? That last set is small by construction, and it is the closest
this dataset can come to naming the residues that carry functional specificity.

Method
------
Pairs come from the structural search, which reports both a TM-score (fold similarity) and a
sequence identity for the same alignment - so both axes are measured on the same
correspondence rather than on two independent comparisons. Thresholds are the conventional
ones: TM-score 0.5 for a shared fold, 30% identity for reliable sequence homology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from structure_analysis.structural_search import StructuralHit

# TM-score at or above which two proteins share a fold. The conventional threshold.
SAME_FOLD = 0.5

# TM-score above which the fold is not merely shared but closely conserved.
CONSERVED_FOLD = 0.7

# Sequence identity above which pairwise homology is reliable; below it lies the twilight
# zone where sequence methods fail and structure does not.
CONSERVED_SEQUENCE = 0.30

# Identity below which sequence has genuinely diverged rather than merely drifted.
DIVERGED_SEQUENCE = 0.20


@dataclass(frozen=True, slots=True)
class CoupledPair:
    """One protein pair, classified by how sequence and structure diverged together."""

    query: str
    target: str
    sequence_identity: float
    tm_score: float
    shared_function: bool | None = None

    @property
    def regime(self) -> str:
        """Which of the four sequence-structure regimes this pair falls in."""
        conserved_sequence = self.sequence_identity >= CONSERVED_SEQUENCE
        conserved_structure = self.tm_score >= CONSERVED_FOLD
        if conserved_sequence and conserved_structure:
            return "both_conserved"
        if not conserved_sequence and conserved_structure:
            return "sequence_diverged_structure_conserved"
        if conserved_sequence and not conserved_structure:
            return "sequence_conserved_structure_diverged"
        return "both_diverged"

    @property
    def is_informative(self) -> bool:
        """Whether the pair is one of the two regimes worth studying.

        Pairs where everything changed together, or nothing did, carry no information
        about which changes matter.
        """
        return self.regime in {
            "sequence_diverged_structure_conserved",
            "sequence_conserved_structure_diverged",
        }

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "query": self.query,
            "target": self.target,
            "sequence_identity": round(self.sequence_identity, 4),
            "tm_score": round(self.tm_score, 4),
            "regime": self.regime,
            "shared_function": self.shared_function,
        }


def classify_pairs(
    hits: Sequence[StructuralHit],
    *,
    function_terms: Mapping[str, Sequence[str]] | None = None,
) -> list[CoupledPair]:
    """Classify every structural pair by how sequence and structure diverged.

    When functional terms are supplied, a pair is marked as sharing function if the two
    proteins have any term in common. That is a coarse test - two chaperones both annotated
    "protein folding" share a term without sharing a role - so it is reported alongside the
    regime rather than used to define it.
    """
    pairs: list[CoupledPair] = []
    for hit in hits:
        if hit.tm_score < SAME_FOLD:
            continue
        shared: bool | None = None
        if function_terms is not None:
            query_terms = set(function_terms.get(hit.query, ()))
            target_terms = set(function_terms.get(hit.target, ()))
            shared = bool(query_terms & target_terms) if query_terms and target_terms else None
        pairs.append(
            CoupledPair(
                query=hit.query,
                target=hit.target,
                sequence_identity=hit.sequence_identity,
                tm_score=hit.tm_score,
                shared_function=shared,
            ),
        )
    return pairs


def coupling_report(pairs: Sequence[CoupledPair], *, max_examples: int = 20) -> dict[str, object]:
    """Summarise how often sequence and structure diverge together, and what it means."""
    if not pairs:
        return {"n_pairs": 0, "note": "no structural pairs above the fold threshold"}

    regimes: dict[str, list[CoupledPair]] = {}
    for pair in pairs:
        regimes.setdefault(pair.regime, []).append(pair)

    def function_rate(items: Sequence[CoupledPair]) -> float | None:
        judged = [item for item in items if item.shared_function is not None]
        if not judged:
            return None
        return round(sum(1 for item in judged if item.shared_function) / len(judged), 4)

    permissive = regimes.get("sequence_diverged_structure_conserved", [])
    decisive = regimes.get("sequence_conserved_structure_diverged", [])

    return {
        "n_pairs": len(pairs),
        "regimes": {
            name: {
                "n_pairs": len(items),
                "share": round(len(items) / len(pairs), 4),
                "median_identity": round(sorted(i.sequence_identity for i in items)[len(items) // 2], 4),
                "median_tm_score": round(sorted(i.tm_score for i in items)[len(items) // 2], 4),
                "fraction_sharing_function": function_rate(items),
            }
            for name, items in sorted(regimes.items(), key=lambda kv: -len(kv[1]))
        },
        "examples": {
            "sequence_diverged_structure_conserved": [
                item.to_json_dict() for item in sorted(permissive, key=lambda i: i.sequence_identity)[:max_examples]
            ],
            "sequence_conserved_structure_diverged": [
                item.to_json_dict() for item in sorted(decisive, key=lambda i: i.tm_score)[:max_examples]
            ],
        },
        "interpretation": (
            "Pairs whose sequence diverged while the fold held mark positions the structure "
            "tolerated changing. Pairs whose sequence held while the fold moved are the rarer "
            "and more informative case: few substitutions did what many elsewhere did not, so "
            "those substitutions are candidates for the ones that carry function. Comparing "
            "the fraction of each regime that still shares function is what turns the "
            "geometry into a statement about which changes matter."
        ),
    }


def regime_by_group(
    pairs: Sequence[CoupledPair],
    group_of: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Regime counts per subpopulation, keyed by the group each query belongs to.

    A subpopulation under looser structural constraint should show more pairs whose sequence
    diverged while the fold held; one under tighter constraint should show fewer.
    """
    counts: dict[str, dict[str, int]] = {}
    for pair in pairs:
        group = group_of.get(pair.query)
        if group is None:
            continue
        counts.setdefault(group, {}).setdefault(pair.regime, 0)
        counts[group][pair.regime] += 1
    return counts
