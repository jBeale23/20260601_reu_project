"""Does the grammar of disordered regions say something the folded regions do not?

The grammar features carried through the rest of this project are computed on whole
sequences. Only three of roughly fifty are region-specific, and those three are summaries -
a count, a mean, a maximum. Everything informative about how a protein is written - the
entropy spectrum across n-mer orders, conditional complexity along the chain, charge
segregation - is measured over the whole protein, which averages an intrinsically
disordered linker against a folded domain and reports one number for both.

That is the wrong unit for this family. Roughly half of J-domain protein sequence is
unalignable: 32.76M residues route to alignment and 31.83M to alignment-free comparison.
A measure averaged across that boundary is describing two different kinds of sequence at
once, and the resulting number belongs to neither.

What separating them makes testable
-----------------------------------
Three questions that the pooled features cannot ask:

*Do the two compartments differ grammatically at all?* They should - disordered sequence is
compositionally biased and low-complexity by definition - and if they do not, the routing
is not doing what it claims.

*Which compartment carries the class signal?* If architecture-based classification works
because of what the folded domains look like, the structured compartment should carry it.
If it works because of the linkers, the disordered compartment should. These are different
claims about why the classifier works, and the pooled features cannot distinguish them.

*Does disordered grammar predict structure and function independently?* An IDR has no fold
to measure, so any association between its grammar and a structural property of the whole
protein is a statement about how disorder constrains the rest of the chain - which is a
different and more interesting claim than the near-tautology that low-complexity sequence
is disordered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.regions import ROUTE_MSA, ROUTE_SHARK
from experimental.grammar_classifier import grammar_features

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.pipeline import ProteinLayout

logger = logging.getLogger(__name__)

DISORDERED = "disordered"
STRUCTURED = "structured"
WHOLE = "whole"
COMPARTMENTS = (DISORDERED, STRUCTURED, WHOLE)

# Shortest concatenated compartment worth measuring. Below this the entropy spectrum is
# dominated by how few residues there are rather than by how they are arranged.
MIN_COMPARTMENT_LENGTH = 40


@dataclass(frozen=True, slots=True)
class CompartmentGrammar:
    """Grammar features for one protein, split by where in the chain they were measured."""

    accession: str
    disordered: dict[str, float]
    structured: dict[str, float]
    whole: dict[str, float]
    n_disordered_residues: int
    n_structured_residues: int

    @property
    def disordered_fraction(self) -> float:
        """Share of measured residues that are disordered."""
        total = self.n_disordered_residues + self.n_structured_residues
        return self.n_disordered_residues / total if total else 0.0

    def by_compartment(self, compartment: str) -> dict[str, float]:
        """Features for one named compartment."""
        return {
            DISORDERED: self.disordered,
            STRUCTURED: self.structured,
            WHOLE: self.whole,
        }.get(compartment, {})


def compartment_sequences(layout: ProteinLayout) -> tuple[str, str]:
    """Concatenated disordered and structured sequence for one protein.

    Concatenated rather than measured per region and averaged: the entropy of a 300-residue
    linker is a different quantity from the mean entropy of six 50-residue pieces, and the
    former is what "how is this protein's disordered sequence written" means.
    """
    disordered = "".join(r.sequence for r in layout.regions if r.route == ROUTE_SHARK and r.sequence)
    structured = "".join(r.sequence for r in layout.regions if r.route == ROUTE_MSA and r.sequence)
    return disordered, structured


def grammar_by_compartment(layout: ProteinLayout) -> CompartmentGrammar | None:
    """Compute the full grammar feature set separately for each compartment.

    Returns ``None`` when either compartment is too short to measure, because a comparison
    needs both sides. Proteins that are entirely disordered or entirely folded are real and
    interesting, but they cannot answer a question about the difference between the two.
    """
    disordered, structured = compartment_sequences(layout)
    if len(disordered) < MIN_COMPARTMENT_LENGTH or len(structured) < MIN_COMPARTMENT_LENGTH:
        return None
    return CompartmentGrammar(
        accession=layout.record.accession,
        disordered=grammar_features(disordered),
        structured=grammar_features(structured),
        whole=grammar_features(layout.record.sequence),
        n_disordered_residues=len(disordered),
        n_structured_residues=len(structured),
    )


def compartment_divergence(grammars: Sequence[CompartmentGrammar]) -> dict[str, float]:
    """How far each feature differs between the two compartments, as a median gap.

    A sanity check before anything else is claimed: if the two compartments were
    grammatically indistinguishable, the routing that produced them would not be separating
    anything, and every downstream comparison would be measuring noise.
    """
    if not grammars:
        return {}
    names = sorted(set(grammars[0].disordered) & set(grammars[0].structured))
    divergence: dict[str, float] = {}
    for name in names:
        gaps = [
            item.disordered[name] - item.structured[name]
            for item in grammars
            if name in item.disordered and name in item.structured
        ]
        if gaps:
            gaps.sort()
            divergence[name] = gaps[len(gaps) // 2]
    return divergence


def compartment_report(
    grammars: Sequence[CompartmentGrammar],
    *,
    top: int = 20,
) -> dict[str, object]:
    """What separating the compartments reveals."""
    if not grammars:
        return {"n_proteins": 0, "note": "no protein had both compartments long enough to measure"}

    divergence = compartment_divergence(grammars)
    ranked = sorted(divergence.items(), key=lambda item: -abs(item[1]))
    fractions = sorted(item.disordered_fraction for item in grammars)

    return {
        "n_proteins": len(grammars),
        "median_disordered_fraction": round(fractions[len(fractions) // 2], 4),
        "median_disordered_residues": sorted(item.n_disordered_residues for item in grammars)[len(grammars) // 2],
        "median_structured_residues": sorted(item.n_structured_residues for item in grammars)[len(grammars) // 2],
        "most_divergent_features": [
            {"feature": name, "median_disordered_minus_structured": round(value, 4)} for name, value in ranked[:top]
        ],
        "note": (
            "Features are computed on the concatenated sequence of each compartment, not "
            "averaged over regions: the entropy of one 300-residue linker is a different "
            "quantity from the mean entropy of six 50-residue pieces, and the first is what "
            "'how is this protein's disordered sequence written' actually means. Proteins "
            "lacking either compartment are excluded, since a comparison needs both sides - "
            "which means fully disordered and fully folded proteins are absent here and "
            "must be looked at separately."
        ),
    }


def features_for_correlation(
    grammars: Sequence[CompartmentGrammar],
    compartment: str,
) -> dict[str, Mapping[str, float]]:
    """One compartment's features, keyed by accession, ready for correlation."""
    if compartment not in COMPARTMENTS:
        message = f"unknown compartment {compartment!r}; expected one of {COMPARTMENTS}"
        raise ValueError(message)
    return {item.accession: item.by_compartment(compartment) for item in grammars}
