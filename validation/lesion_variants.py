"""Do grammatical lesions coincide with pathogenic variants?

This is the test that decides whether reading a protein as a sentence is a real idea or a
metaphor. :mod:`domain_layout.syntax_errors` finds spans that read badly and types them;
that machinery is validated only on lesions constructed by hand, which proves it can find
what it was built to find and nothing more. If the spans mean anything biologically,
variants that break the protein should fall inside them more often than variants that do
not.

The control is benign variants, not shuffled positions
------------------------------------------------------
A positional permutation asks whether lesions overlap variants more than a random stretch
of the same protein would. That leaves every real confound standing: variants are not
uniformly distributed, they concentrate in exons that get sequenced, in domains that get
studied, and in regions where substitutions are tolerated enough to be catalogued at all.

Benign variants share all of it - the same gene, the same sequencing depth, the same
clinical attention, the same domain coverage - and differ in exactly the thing under test.
Comparing pathogenic against benign, rather than pathogenic against a shuffle, is what
turns this from a positional statistic into a claim about pathogenicity.

Reference-residue matching
--------------------------
ClinVar numbers a change against a transcript, and reports one per isoform. A change is
only used when its reference residue matches the sequence held here at that position. That
resolves the isoform and validates the mapping in one step: a change matching nothing is
numbered against a protein this analysis does not have, and counting it would place a real
variant at an arbitrary position.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from data_fetching.fetch_clinvar import VariantPosition
    from domain_layout.syntax_errors import GrammaticalLesion

logger = logging.getLogger(__name__)

SIGNIFICANCE_LEVEL = 0.05

# Below this many usable variants of either class, the 2x2 table is too sparse to test.
MIN_VARIANTS_PER_CLASS = 5


@dataclass(frozen=True, slots=True)
class Coincidence:
    """Whether pathogenic variants fall in lesions more than benign ones do."""

    scope: str
    n_pathogenic_in: int
    n_pathogenic_out: int
    n_benign_in: int
    n_benign_out: int
    p_value: float
    p_adjusted: float = 1.0

    @property
    def pathogenic_rate(self) -> float:
        """Share of pathogenic variants inside a lesion."""
        total = self.n_pathogenic_in + self.n_pathogenic_out
        return self.n_pathogenic_in / total if total else 0.0

    @property
    def benign_rate(self) -> float:
        """Share of benign variants inside a lesion."""
        total = self.n_benign_in + self.n_benign_out
        return self.n_benign_in / total if total else 0.0

    @property
    def odds_ratio(self) -> float:
        """Odds of a variant being pathogenic inside a lesion versus outside.

        Haldane-corrected, so a zero cell gives a finite ratio rather than an infinity that
        would dominate any ranking.
        """
        a = self.n_pathogenic_in + 0.5
        b = self.n_pathogenic_out + 0.5
        c = self.n_benign_in + 0.5
        d = self.n_benign_out + 0.5
        return (a * d) / (b * c)

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "scope": self.scope,
            "n_pathogenic_in_lesion": self.n_pathogenic_in,
            "n_pathogenic_outside": self.n_pathogenic_out,
            "n_benign_in_lesion": self.n_benign_in,
            "n_benign_outside": self.n_benign_out,
            "pathogenic_rate_in_lesion": round(self.pathogenic_rate, 4),
            "benign_rate_in_lesion": round(self.benign_rate, 4),
            "odds_ratio": round(self.odds_ratio, 3),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
        }


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for a 2x2 table."""
    total = a + b + c + d
    if total == 0:
        return 1.0
    row1, col1 = a + b, a + c
    denominator = comb(total, col1)
    if denominator == 0:
        return 1.0
    observed = comb(row1, a) * comb(total - row1, col1 - a) if 0 <= col1 - a <= total - row1 else 0
    if observed == 0:
        return 1.0
    low = max(0, col1 - (total - row1))
    high = min(row1, col1)
    tail = 0
    for value in range(low, high + 1):
        weight = comb(row1, value) * comb(total - row1, col1 - value)
        if weight <= observed:
            tail += weight
    return min(1.0, tail / denominator)


def usable_variants(
    variants: Sequence[VariantPosition],
    sequence: str,
) -> list[VariantPosition]:
    """Variants whose reference residue matches the sequence at their position.

    This is the isoform check. ClinVar reports one change per transcript, so a variant may
    carry several positions of which at most one refers to the protein held here.
    """
    usable: list[VariantPosition] = []
    for variant in variants:
        index = variant.position - 1
        if 0 <= index < len(sequence) and sequence[index] == variant.reference:
            usable.append(variant)
    return usable


def _in_lesion(position: int, lesions: Sequence[GrammaticalLesion]) -> bool:
    """Whether a 1-based residue position falls inside any lesion span."""
    return any(lesion.start <= position <= lesion.end for lesion in lesions)


def coincidence_by_scope(
    per_protein: Mapping[str, tuple[str, Sequence[VariantPosition], Sequence[GrammaticalLesion]]],
) -> list[Coincidence]:
    """Test overall, and once per lesion type.

    Per type as well as overall, because the types are different hypotheses. A hydrophobic
    exposure and a low-complexity linker are both "lesions" and only one of them is an
    obvious candidate for pathogenicity; pooling them could hide a real effect in one behind
    noise from another.
    """
    from domain_layout.syntax_errors import ERROR_TYPES  # noqa: PLC0415 - avoids a cycle

    scopes: dict[str, list[int]] = {"any_lesion": [0, 0, 0, 0]}
    for error_type in ERROR_TYPES:
        scopes[error_type] = [0, 0, 0, 0]

    for sequence, variants, lesions in per_protein.values():
        usable = usable_variants(variants, sequence)
        for variant in usable:
            inside_any = _in_lesion(variant.position, lesions)
            slot = 0 if variant.is_pathogenic else 2
            scopes["any_lesion"][slot + (0 if inside_any else 1)] += 1
            for error_type in ERROR_TYPES:
                typed = [item for item in lesions if item.error_type == error_type]
                inside = _in_lesion(variant.position, typed)
                scopes[error_type][slot + (0 if inside else 1)] += 1

    results: list[Coincidence] = []
    for scope, (p_in, p_out, b_in, b_out) in scopes.items():
        if p_in + p_out < MIN_VARIANTS_PER_CLASS or b_in + b_out < MIN_VARIANTS_PER_CLASS:
            continue
        results.append(
            Coincidence(
                scope=scope,
                n_pathogenic_in=p_in,
                n_pathogenic_out=p_out,
                n_benign_in=b_in,
                n_benign_out=b_out,
                p_value=_fisher_two_sided(p_in, p_out, b_in, b_out),
            ),
        )

    adjusted = benjamini_hochberg({item.scope: item.p_value for item in results})
    return [
        Coincidence(
            scope=item.scope,
            n_pathogenic_in=item.n_pathogenic_in,
            n_pathogenic_out=item.n_pathogenic_out,
            n_benign_in=item.n_benign_in,
            n_benign_out=item.n_benign_out,
            p_value=item.p_value,
            p_adjusted=adjusted[item.scope],
        )
        for item in results
    ]


def coincidence_report(
    per_protein: Mapping[str, tuple[str, Sequence[VariantPosition], Sequence[GrammaticalLesion]]],
) -> dict[str, object]:
    """The test that decides whether the lesion idea survives contact with real variants."""
    results = coincidence_by_scope(per_protein)
    overall = next((item for item in results if item.scope == "any_lesion"), None)

    matched = 0
    dropped = 0
    lesion_types: Counter[str] = Counter()
    for sequence, variants, lesions in per_protein.values():
        usable = usable_variants(variants, sequence)
        matched += len(usable)
        dropped += len(variants) - len(usable)
        for lesion in lesions:
            lesion_types[lesion.error_type] += 1

    return {
        "n_proteins": len(per_protein),
        "n_variants_matched_to_sequence": matched,
        "n_variants_dropped_isoform_mismatch": dropped,
        "lesions_by_type": dict(lesion_types.most_common()),
        "n_scopes_tested": len(results),
        "n_significant": sum(1 for item in results if item.p_adjusted < SIGNIFICANCE_LEVEL),
        "overall": overall.to_json_dict() if overall else None,
        "by_scope": [item.to_json_dict() for item in results],
        "interpretation": (
            "Benign variants are the control, not shuffled positions. They share the gene, "
            "the sequencing depth, the clinical attention and the domain coverage of the "
            "pathogenic set, and differ in exactly the thing under test - so an odds ratio "
            "above one is a statement about pathogenicity rather than about where variants "
            "happen to be catalogued. A ratio near one means the lesions, whatever else "
            "they describe, do not mark where a protein breaks."
        ),
    }
