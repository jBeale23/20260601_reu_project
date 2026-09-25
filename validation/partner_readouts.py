"""Shared partner readouts, and what a candidate partition is doing to the reference.

Two cluster drivers - the partition search and the class-sufficiency test - both reduce the
BioGRID interaction map to the same three readouts, and their results are only comparable if
they reduce it *identically*. Kept in two places those definitions drift, and the drift is
invisible: both scripts still run, both still print a Cramer's V, and the two numbers quietly
stop being about the same thing. They live here so there is one definition.

The module also answers a question the search itself cannot. A candidate partition that
scores better than A/B/C has not necessarily found a new way to divide the proteins - it may
be A/B/C with two classes merged, or one class split. Those mean opposite things:

- A **coarsening** that scores better says one of the existing boundaries carries no signal
  for that readout. The finding is about which boundary to *drop*.
- A **refinement** that scores better says an existing class hides real structure. The
  finding is about which class to *split*.
- A **relabeling** is the same partition under different names and can never be a finding,
  however good its score.
- A **crossing** partition is genuinely new: it groups proteins the reference separates and
  separates proteins it groups.

Without this distinction ``has_zinc_finger_like`` reads as a discovery, when in fact every
protein carrying that feature is class A by definition and the partition is exactly
"A against everything else".
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# An interactome with at least this share of Hsp70 partners counts as Hsp70-rich. Low
# because, once the non-chaperone false positives are excluded, real JDP interactomes carry
# only a few percent Hsp70 - see the corrected fractions in the pairing sweep.
HSP70_RICH_THRESHOLD = 0.01

# How to handle a protein whose partners have no single commonest label.
DROP_TIES = "drop"
FIRST_ON_TIE = "first"

RELABELING = "relabeling"
REFINEMENT = "refinement"
COARSENING = "coarsening"
CROSSING = "crossing"


def dominant_label(
    interactions: Mapping[str, Sequence[str]],
    mapper: Callable[[str], str | None],
    *,
    on_tie: str = DROP_TIES,
) -> dict[str, str]:
    """The commonest non-empty label among each protein's partners.

    Proteins whose partners all map to nothing are absent rather than present with a
    placeholder, so an "unassigned" bucket cannot grow large enough to dominate a
    contingency table on its own.

    Ties are the subtle part. On the Hsp70 paralogue readout **23.6% of proteins have no
    single commonest partner class** - two or more tie, often three-way. The original
    implementation resolved these with ``max(set(labels), key=labels.count)``, and iterating
    a set of strings is order-unstable under hash randomisation, so the tie broke differently
    from run to run. That moved Cramer's V by about 0.13 between otherwise identical runs,
    which is larger than most of the effects being compared.

    The default is therefore to drop tied proteins: a protein whose partners split evenly
    does not *have* a dominant partner, and assigning it one invents an observation. Pass
    ``on_tie="first"`` for a deterministic arbitrary pick when a stable label is needed for
    every protein, but be aware that it is a convention, not a measurement.

    Args:
        interactions: Partner gene symbols per protein.
        mapper: Gene symbol to label, or ``None`` for "not in this scheme".
        on_tie: ``drop`` to exclude ambiguous proteins, ``first`` to take the
            lexicographically first tied label.

    Returns:
        Label per protein, for proteins that have an unambiguous one.
    """
    if on_tie not in {DROP_TIES, FIRST_ON_TIE}:
        msg = f"on_tie must be {DROP_TIES!r} or {FIRST_ON_TIE!r}, got {on_tie!r}"
        raise ValueError(msg)

    out: dict[str, str] = {}
    for accession, genes in interactions.items():
        labels = [label for label in (mapper(gene) for gene in genes) if label and label != "unassigned"]
        if not labels:
            continue
        counts = Counter(labels)
        best = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == best)
        if len(winners) > 1 and on_tie == DROP_TIES:
            continue
        out[accession] = winners[0]
    return out


def tie_rate(
    interactions: Mapping[str, Sequence[str]],
    mapper: Callable[[str], str | None],
) -> dict[str, float]:
    """How much of a readout is decided by an arbitrary tie-break.

    Reported alongside any result computed from a dominant label, because a readout that is
    a quarter ties is a much weaker measurement than its sample size suggests.
    """
    labelled = tied = 0
    for genes in interactions.values():
        labels = [label for label in (mapper(gene) for gene in genes) if label and label != "unassigned"]
        if not labels:
            continue
        labelled += 1
        counts = Counter(labels)
        best = max(counts.values())
        if sum(1 for count in counts.values() if count == best) > 1:
            tied += 1
    return {
        "n_labelled": labelled,
        "n_tied": tied,
        "tie_fraction": round(tied / labelled, 4) if labelled else 0.0,
    }


def interactome_scope(
    interactions: Mapping[str, Sequence[str]],
    is_hsp70: Callable[[str], bool],
    *,
    threshold: float = HSP70_RICH_THRESHOLD,
) -> dict[str, str]:
    """Split proteins by whether their interactome is Hsp70-rich or Hsp70-poor."""
    return {
        accession: ("hsp70_rich" if sum(1 for g in genes if is_hsp70(g)) / len(genes) >= threshold else "hsp70_poor")
        for accession, genes in interactions.items()
        if genes
    }


def partition_relationship(candidate: Mapping[str, str], reference: Mapping[str, str]) -> str:
    """How a candidate partition sits against the reference, on the proteins they share.

    Returns one of ``relabeling``, ``refinement``, ``coarsening``, or ``crossing``. A
    partition that is a relabeling cannot be a discovery no matter how it scores, and one
    that is a coarsening is a statement about a boundary to remove rather than a new scheme.
    """
    shared = [a for a in candidate if a in reference]
    if not shared:
        return CROSSING

    # A candidate group is "pure" when everything in it shares one reference class, and vice
    # versa. Pure both ways is a relabeling; pure one way tells you which direction it went.
    by_candidate: dict[str, set[str]] = {}
    by_reference: dict[str, set[str]] = {}
    for accession in shared:
        by_candidate.setdefault(candidate[accession], set()).add(reference[accession])
        by_reference.setdefault(reference[accession], set()).add(candidate[accession])

    candidate_pure = all(len(values) == 1 for values in by_candidate.values())
    reference_pure = all(len(values) == 1 for values in by_reference.values())

    if candidate_pure and reference_pure:
        return RELABELING
    if candidate_pure:
        # Every candidate group sits inside one reference class: the candidate splits.
        return REFINEMENT
    if reference_pure:
        # Every reference class sits inside one candidate group: the candidate merges.
        return COARSENING
    return CROSSING


def describe_relationship(relationship: str) -> str:
    """One sentence on what a relationship implies for a candidate that scored well."""
    return {
        RELABELING: (
            "the same partition under different names, so a better score here is a naming artefact and not a finding"
        ),
        REFINEMENT: ("a split of the existing classes, so a better score says one class hides real structure"),
        COARSENING: (
            "a merge of the existing classes, so a better score says one of the existing "
            "boundaries carries no signal for this readout"
        ),
        CROSSING: ("genuinely new: it groups proteins the reference separates and separates proteins it groups"),
    }.get(relationship, "an unrecognised relationship to the reference")
