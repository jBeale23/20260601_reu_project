"""Redundancy control by sequence-identity clustering, on three region definitions.

Why this exists
---------------
Cross-validation over homologous proteins reports how well a method recognises relatives
it has already seen, which is not the question. Genus blocking was the first attempt and it
did almost nothing: removing 2,174 same-genus relatives changed accuracy not at all, because
122 of 129 labelled proteins sit in *cross*-genus ortholog groups. Taxonomy is the wrong
unit. Sequence identity is the standard one, and the standard threshold is 30%.

Three clusterings, not one
--------------------------
A J-domain protein is a conserved J-domain bolted to a variable remainder, so a single
whole-sequence clustering answers a blurred question. Clustering is therefore done three
ways and reported side by side:

``full_sequence``
    The whole protein. Closest to convention, but dominated by whichever part is longer.
``j_domain``
    The J-domain regions only. These are near-identical across the family, so this is the
    *strictest* partition - almost everything collapses together, and a method that
    survives it is not leaning on J-domain similarity at all.
``non_j_domain``
    Everything except the J-domain. This is where architecture actually varies, so it is
    the partition that matches what the classifier claims to use.

They disagree by construction, and the disagreement is the point: a result that holds under
all three is not an artefact of how redundancy was defined, and one that holds under only
the loosest is.

Method
------
Greedy incremental clustering, the same rule CD-HIT uses: sequences are taken longest-first,
each either joins the first representative it matches at or above the threshold, or becomes
a representative itself. Identity is computed from a global alignment and normalised by the
shorter sequence, so a short fragment inside a long protein counts as identical rather than
as 30% of it.

The all-pairs cost is quadratic, which is fine at the scale that matters here - the labelled
evaluation set is ~129 proteins, or ~8,000 comparisons. A k-mer prefilter skips alignments
that cannot possibly reach the threshold, which is what keeps it usable on larger inputs;
for proteome-scale work MMseqs2 remains the right tool and this is not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from Bio.Align import PairwiseAligner, substitution_matrices

from domain_layout.constants import FAMILY_J_DOMAIN

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from domain_layout.pipeline import ProteinLayout

# The conventional redundancy threshold for structural and functional benchmarks. Below
# roughly this, two proteins are no longer assumed to share function by descent.
DEFAULT_IDENTITY_THRESHOLD = 0.30

# Word size for the prefilter. Two sequences sharing fewer than the implied number of
# k-mers cannot reach the threshold, so the alignment is skipped.
PREFILTER_WORD_SIZE = 3

REGION_FULL = "full_sequence"
REGION_J_DOMAIN = "j_domain"
REGION_NON_J = "non_j_domain"
REGIONS = (REGION_FULL, REGION_J_DOMAIN, REGION_NON_J)

# Shortest region worth clustering. Below this, identity is dominated by chance and a
# handful of residues decides the cluster.
MIN_REGION_LENGTH = 20

# Minimum length ratio for two sequences to be considered redundant, shorter over longer.
#
# Identity alone is not enough, and leaving it out collapsed the entire labelled set into a
# single cluster. Identity here is normalised by the shorter sequence - correct for asking
# "is this fragment redundant with that protein" - but under greedy clustering it lets the
# single longest region absorb every shorter one that shares a 30%-identical stretch,
# however small a part of the long protein that stretch is. CD-HIT pairs its identity
# threshold with a length requirement for exactly this reason. At 0.5, a region less than
# half the length of a representative starts its own cluster instead, which is the
# conservative direction: it yields more clusters, hence more independent folds.
MIN_LENGTH_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """Cluster membership for one region definition."""

    region: str
    threshold: float
    representative_of: dict[str, str]
    n_sequences: int
    n_clusters: int
    n_too_short: int

    @property
    def reduction(self) -> float:
        """Fraction of sequences removed by collapsing each cluster to one member."""
        if self.n_sequences == 0:
            return 0.0
        return 1.0 - (self.n_clusters / self.n_sequences)

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "region": self.region,
            "identity_threshold": self.threshold,
            "n_sequences": self.n_sequences,
            "n_clusters": self.n_clusters,
            "n_too_short_to_cluster": self.n_too_short,
            "redundancy_reduction": round(self.reduction, 4),
        }


def _build_aligner() -> PairwiseAligner:
    """A global aligner with BLOSUM62 and affine gaps.

    The gap penalties are not decoration. An earlier version scored matches only and left
    both gap scores at zero, which makes gaps free - the aligner then maximises matches by
    inserting unlimited gaps, and what it computes is the longest common subsequence rather
    than an alignment. Over a twenty-letter alphabet that scores two *random* sequences at
    0.33-0.35 identity, above the 30% threshold, so every protein clustered with every
    other and the whole partition collapsed to one cluster.

    -11 to open and -1 to extend are the BLAST/BLOSUM62 conventions, which is what makes an
    identity computed here comparable to one quoted from any standard tool.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def _words(sequence: str, size: int = PREFILTER_WORD_SIZE) -> set[str]:
    """The set of k-mers in a sequence."""
    return {sequence[index : index + size] for index in range(len(sequence) - size + 1)}


def sequence_identity(first: str, second: str, *, aligner: PairwiseAligner | None = None) -> float:
    """Identity between two sequences, normalised by the shorter one.

    Identity is counted from the alignment itself - positions where the two aligned
    residues are the same character - rather than derived from the alignment score. A score
    conflates similarity with identity, because BLOSUM62 credits conservative substitutions,
    and a redundancy threshold is conventionally about identity.

    Normalising by the shorter sequence rather than the alignment length is deliberate: a
    J-domain that is a complete substring of a large multi-domain protein is 100% identical
    to it for redundancy purposes, and dividing by the longer length would call that pair
    distant and leave both in the evaluation set.
    """
    if not first or not second:
        return 0.0
    shortest = min(len(first), len(second))
    if shortest == 0:
        return 0.0
    engine = aligner if aligner is not None else _build_aligner()
    try:
        alignment = next(iter(engine.align(first, second)))
    except (StopIteration, ValueError):
        return 0.0
    top, bottom = str(alignment[0]), str(alignment[1])
    identical = sum(1 for left, right in zip(top, bottom, strict=False) if left == right and left != "-")
    return min(1.0, identical / shortest)


def _can_reach(first: str, second: str, threshold: float) -> bool:
    """Cheap k-mer bound: whether the pair could possibly reach the threshold."""
    if len(first) < PREFILTER_WORD_SIZE or len(second) < PREFILTER_WORD_SIZE:
        return True
    shared = len(_words(first) & _words(second))
    shortest = min(len(first), len(second))
    # An identity of i over the shorter sequence implies at least i*L - (k-1) shared words
    # in the ideal case; requiring a fraction of that is a safe, loose filter.
    needed = max(1, int(threshold * (shortest - PREFILTER_WORD_SIZE + 1) * 0.5))
    return shared >= needed


def greedy_clusters(
    sequences: Mapping[str, str],
    *,
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    min_length_ratio: float = MIN_LENGTH_RATIO,
) -> dict[str, str]:
    """Cluster sequences, returning each accession's representative.

    Longest-first, as CD-HIT does: the longest sequence in a cluster becomes its
    representative, which keeps the retained sequence the most informative one rather than
    whichever happened to be seen first.
    """
    aligner = _build_aligner()
    ordered = sorted(sequences, key=lambda key: (-len(sequences[key]), key))
    representatives: list[str] = []
    assignment: dict[str, str] = {}
    for accession in ordered:
        sequence = sequences[accession]
        for representative in representatives:
            candidate = sequences[representative]
            # Length first: it is far cheaper than an alignment, and without it one long
            # region absorbs the whole set.
            ratio = min(len(sequence), len(candidate)) / max(len(sequence), len(candidate))
            if ratio < min_length_ratio:
                continue
            if not _can_reach(sequence, candidate, threshold):
                continue
            if sequence_identity(sequence, candidate, aligner=aligner) >= threshold:
                assignment[accession] = representative
                break
        else:
            representatives.append(accession)
            assignment[accession] = accession
    return assignment


def region_sequences(layout: ProteinLayout, region: str) -> str:
    """The sequence of one region definition for a protein.

    ``non_j_domain`` concatenates everything outside the J-domain rather than taking the
    longest piece, because a protein's non-J character is the whole remainder - splitting
    it would discard exactly the architecture the classifier is claimed to read.
    """
    if region == REGION_FULL:
        return layout.record.sequence
    j_parts = [item.sequence for item in layout.regions if item.family == FAMILY_J_DOMAIN]
    if region == REGION_J_DOMAIN:
        return "".join(j_parts)
    if region == REGION_NON_J:
        return "".join(item.sequence for item in layout.regions if item.family != FAMILY_J_DOMAIN)
    message = f"unknown region definition: {region}"
    raise ValueError(message)


def cluster_layouts(
    layouts: Sequence[ProteinLayout],
    *,
    region: str,
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    restrict_to: Sequence[str] | None = None,
    min_length_ratio: float = MIN_LENGTH_RATIO,
) -> ClusterAssignment:
    """Cluster a set of layouts under one region definition."""
    wanted = set(restrict_to) if restrict_to is not None else None
    sequences: dict[str, str] = {}
    too_short = 0
    for layout in layouts:
        accession = layout.record.accession
        if wanted is not None and accession not in wanted:
            continue
        sequence = region_sequences(layout, region)
        if len(sequence) < MIN_REGION_LENGTH:
            too_short += 1
            continue
        sequences[accession] = sequence

    assignment = greedy_clusters(sequences, threshold=threshold, min_length_ratio=min_length_ratio)
    return ClusterAssignment(
        region=region,
        threshold=threshold,
        representative_of=assignment,
        n_sequences=len(sequences),
        n_clusters=len(set(assignment.values())),
        n_too_short=too_short,
    )


def cluster_blocked_folds(
    assignment: Mapping[str, str],
    accessions: Sequence[str],
    *,
    n_folds: int = 5,
) -> list[list[str]]:
    """Split accessions into folds that never split a cluster across the boundary.

    This is what makes the evaluation honest: if two proteins are more than ``threshold``
    identical, they are in the same fold, so a method is never tested on a near-copy of
    something it trained on.
    """
    by_cluster: dict[str, list[str]] = {}
    for accession in accessions:
        by_cluster.setdefault(assignment.get(accession, accession), []).append(accession)

    # Largest clusters placed first, into whichever fold is currently smallest, so no fold
    # ends up holding most of the data.
    folds: list[list[str]] = [[] for _ in range(max(1, n_folds))]
    for members in sorted(by_cluster.values(), key=len, reverse=True):
        smallest = min(folds, key=len)
        smallest.extend(sorted(members))
    return folds


def clustering_report(
    layouts: Sequence[ProteinLayout],
    *,
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    restrict_to: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cluster under all three region definitions and report them together.

    The three are reported side by side rather than reduced to one number, because they
    answer different questions and a reader needs to see where they disagree.
    """
    assignments = {
        region: cluster_layouts(layouts, region=region, threshold=threshold, restrict_to=restrict_to)
        for region in REGIONS
    }
    return {
        "identity_threshold": threshold,
        "method": (
            "Greedy incremental clustering, longest sequence first, identity from a global "
            "alignment normalised by the shorter sequence. Equivalent in rule to CD-HIT; "
            "MMseqs2 remains preferable at proteome scale."
        ),
        "by_region": {region: item.to_json_dict() for region, item in assignments.items()},
        "note": (
            "The three partitions disagree by construction. j_domain is the strictest - "
            "J-domains are near-identical family-wide, so it collapses almost everything "
            "and a method surviving it is not leaning on J-domain similarity. "
            "non_j_domain is the partition that matches what the classifier claims to "
            "read. A result holding under all three is not an artefact of how redundancy "
            "was defined."
        ),
    }
