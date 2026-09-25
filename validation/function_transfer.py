"""Transfer functional annotation by homology, for proteins that have none.

The measurement that forces this: of 400 proteins sampled from the set, 74% carry Gene
Ontology terms but only 0.8% are reviewed, and an unreviewed entry carried *zero* evidence
records. Those terms are electronically inferred, and the commonest route is InterPro2GO -
the domain signature implies the term. Using them to test whether domain architecture
predicts function would be circular, because the answer was assumed when the annotation
was made.

So the experimentally-backed annotation is the only usable evidence, and it exists almost
exclusively for model organisms. Transfer is how a claim about non-model organisms can be
made at all.

Why profile HMMs rather than BLASTP, Dali, or a protein language model
----------------------------------------------------------------------
All four would work. Profiles are chosen because the machinery is already here
(:mod:`domain_layout.hmmer`), so no new heavy dependency enters the pipeline, and because
a profile is more sensitive than pairwise identity at the divergence that matters - which
is the whole reason this project routes half its residues away from alignment.

Dali would be the right choice for a structure-first transfer and would catch homology
that sequence misses entirely. It is worth adding, and the interface here is deliberately
narrow so a structural donor-finder can be substituted without touching the callers.

What transfer is and is not
---------------------------
A transferred term is a **hypothesis**, and every one is labelled with the donor it came
from and the score that justified it. :class:`TransferredAnnotation` is a distinct type
from an observed record precisely so the two cannot be mixed by accident, and any analysis
that uses transferred terms must report them separately from observed ones. A conclusion
that holds only on transferred annotation is a conclusion about the transfer method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout import hmmer as hmmer_backend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from data_fetching.fetch_function import FunctionalRecord

logger = logging.getLogger(__name__)

# Bit score below which a match is not trusted to carry annotation. Profile bit scores are
# comparable across searches of the same profile; 50 is the conventional floor for
# confident homology and corresponds to an E-value far below any usual reporting cut.
MIN_TRANSFER_SCORE = 50.0

# E-value ceiling, applied in addition to the bit score. Both are kept because they fail
# differently: a bit score can be high on a short spurious match, and an E-value depends on
# the size of the database searched.
MAX_TRANSFER_EVALUE = 1e-10

# Donors required to agree before a term transfers. One donor is an anecdote; requiring two
# independent homologs to carry the same term removes most annotation errors in the
# reference set, at the cost of missing terms known from a single organism.
MIN_DONOR_AGREEMENT = 2


@dataclass(frozen=True, slots=True)
class TransferredAnnotation:
    """A functional term inherited from a homolog, never observed directly.

    A separate type from :class:`~data_fetching.fetch_function.FunctionalRecord` so that a
    hypothesis cannot silently be counted as an observation.
    """

    accession: str
    terms: tuple[str, ...]
    donors: tuple[str, ...]
    best_score: float
    best_evalue: float
    n_donors_agreeing: int
    # Which search supplied the donors. A term reached only through structure survived the
    # loss of sequence similarity, which makes it both a stronger claim and the one most
    # dependent on the transfer method being sound.
    route: str = "sequence"

    @property
    def is_transferred(self) -> bool:
        """Always true. Present so callers can branch without checking the type."""
        return True

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "accession": self.accession,
            "terms": list(self.terms),
            "donors": list(self.donors),
            "best_score": round(self.best_score, 2),
            "best_evalue": self.best_evalue,
            "n_donors_agreeing": self.n_donors_agreeing,
            "route": self.route,
            "evidence": f"transferred by {self.route} homology, not observed",
        }


@dataclass(frozen=True, slots=True)
class TransferSummary:
    """Coverage and provenance for one transfer run."""

    n_targets: int
    n_donors_available: int
    n_transferred: int
    n_no_homolog: int
    min_score: float
    min_agreement: int

    @property
    def transfer_rate(self) -> float:
        """Share of targets that received any term."""
        return self.n_transferred / self.n_targets if self.n_targets else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "n_targets": self.n_targets,
            "n_donors_available": self.n_donors_available,
            "n_transferred": self.n_transferred,
            "n_without_homolog": self.n_no_homolog,
            "transfer_rate": round(self.transfer_rate, 4),
            "min_bit_score": self.min_score,
            "min_donor_agreement": self.min_agreement,
            "caveat": (
                "Transferred terms are hypotheses carrying the donor and score that "
                "justified them. Any result computed on them must be reported separately "
                "from results on observed annotation; a conclusion that holds only on "
                "transferred terms is a conclusion about the transfer method."
            ),
        }


def experimental_donors(
    functions: Mapping[str, FunctionalRecord],
    sequences: Mapping[str, str],
) -> dict[str, FunctionalRecord]:
    """Proteins whose annotation is experimentally backed and whose sequence is available.

    Only these may donate. An electronically inferred term would carry its own circularity
    into every protein it was transferred to.
    """
    return {
        accession: record
        for accession, record in functions.items()
        if record.has_experimental_annotation and sequences.get(accession)
    }


@dataclass(frozen=True, slots=True)
class TransferThresholds:
    """How strong a match must be, and how many donors must agree, before a term moves."""

    min_score: float = MIN_TRANSFER_SCORE
    max_evalue: float = MAX_TRANSFER_EVALUE
    min_agreement: int = MIN_DONOR_AGREEMENT


DEFAULT_THRESHOLDS = TransferThresholds()


def _collect_hits(
    targets: Mapping[str, str],
    donors: Mapping[str, FunctionalRecord],
    donor_sequences: Mapping[str, str],
    thresholds: TransferThresholds,
) -> dict[str, list[tuple[str, float, float]]]:
    """Every donor match per target, above the score and E-value floors.

    One profile per donor. A single-sequence profile is still position-specific, which is
    what gives this its sensitivity advantage over pairwise identity.
    """
    # Build every donor profile first, then run one batched search. Searching donor by
    # donor re-encodes all 181,328 targets each time round, which is the whole cost of
    # this function; done once, the searches themselves are cheap and thread across cores.
    built: list[tuple[str, hmmer_backend.FamilyProfile]] = []
    for accession in donors:
        sequence = donor_sequences.get(accession)
        if not sequence:
            continue
        profile = hmmer_backend.build_profile(accession, [sequence, sequence, sequence])
        if profile is None:
            continue
        built.append((accession, profile))

    hits_by_target: dict[str, list[tuple[str, float, float]]] = {}
    if not built:
        return hits_by_target

    results = hmmer_backend.search_profiles(
        [profile for _, profile in built],
        dict(targets),
        e_value_cutoff=thresholds.max_evalue,
    )
    for (accession, _profile), found in zip(built, results, strict=True):
        for target, hit in found.items():
            if hit.score >= thresholds.min_score:
                hits_by_target.setdefault(target, []).append((accession, hit.score, hit.e_value))
    return hits_by_target


def transfer_annotations(
    targets: Mapping[str, str],
    donors: Mapping[str, FunctionalRecord],
    donor_sequences: Mapping[str, str],
    *,
    thresholds: TransferThresholds = DEFAULT_THRESHOLDS,
) -> tuple[dict[str, TransferredAnnotation], TransferSummary]:
    """Infer functional terms for unannotated proteins from their closest annotated homologs.

    Args:
        targets: Accession to sequence, for proteins needing annotation.
        donors: Experimentally annotated records that may donate terms.
        donor_sequences: Accession to sequence for those donors.
        thresholds: Score, E-value, and donor-agreement requirements.

    Returns:
        The transferred annotations and a summary stating coverage and provenance.
    """
    if not targets or not donors or not hmmer_backend.hmmer_available():
        if not hmmer_backend.hmmer_available():
            logger.warning("pyhmmer is not installed; no annotation can be transferred.")
        return {}, TransferSummary(
            n_targets=len(targets),
            n_donors_available=len(donors),
            n_transferred=0,
            n_no_homolog=len(targets),
            min_score=thresholds.min_score,
            min_agreement=thresholds.min_agreement,
        )

    hits_by_target = _collect_hits(targets, donors, donor_sequences, thresholds)

    transferred: dict[str, TransferredAnnotation] = {}
    for target, hits in hits_by_target.items():
        hits.sort(key=lambda item: -item[1])
        donor_ids = [accession for accession, _score, _evalue in hits]

        counts: dict[str, int] = {}
        for accession in donor_ids:
            for term in donors[accession].experimental_terms:
                counts[term] = counts.get(term, 0) + 1
        agreed = tuple(sorted(term for term, count in counts.items() if count >= thresholds.min_agreement))
        if not agreed:
            continue

        transferred[target] = TransferredAnnotation(
            accession=target,
            terms=agreed,
            donors=tuple(donor_ids[:5]),
            best_score=hits[0][1],
            best_evalue=hits[0][2],
            n_donors_agreeing=len(donor_ids),
        )

    return transferred, TransferSummary(
        n_targets=len(targets),
        n_donors_available=len(donors),
        n_transferred=len(transferred),
        n_no_homolog=len(targets) - len(transferred),
        min_score=thresholds.min_score,
        min_agreement=thresholds.min_agreement,
    )


def transfer_from_structure(
    structural_donors: Mapping[str, Sequence[object]],
    donors: Mapping[str, FunctionalRecord],
    *,
    min_agreement: int = MIN_DONOR_AGREEMENT,
) -> dict[str, TransferredAnnotation]:
    """Transfer terms using structural rather than sequence homologs.

    Args:
        structural_donors: Query accession to its structural hits, as produced by
            :func:`structure_analysis.structural_search.best_donors`.
        donors: Experimentally annotated records eligible to donate.
        min_agreement: Donors that must carry a term before it transfers.

    The agreement requirement is the same as for the sequence route, and for the same
    reason: a single donor's annotation error would otherwise propagate unchecked.
    """
    transferred: dict[str, TransferredAnnotation] = {}
    for query, hits in structural_donors.items():
        eligible = [hit for hit in hits if getattr(hit, "target", None) in donors]
        if not eligible:
            continue

        counts: dict[str, int] = {}
        for hit in eligible:
            for term in donors[hit.target].experimental_terms:
                counts[term] = counts.get(term, 0) + 1
        agreed = tuple(sorted(term for term, count in counts.items() if count >= min_agreement))
        if not agreed:
            continue

        best = max(eligible, key=lambda hit: hit.tm_score)
        transferred[query] = TransferredAnnotation(
            accession=query,
            terms=agreed,
            donors=tuple(hit.target for hit in eligible[:5]),
            # TM-score stands in for the bit score on this route; the field name is shared
            # so a reader must check `route` before comparing two numbers.
            best_score=float(best.tm_score),
            best_evalue=float(best.e_value),
            n_donors_agreeing=len(eligible),
            route="structure",
        )
    return transferred


def combine_routes(
    sequence_transfers: Mapping[str, TransferredAnnotation],
    structural_transfers: Mapping[str, TransferredAnnotation],
) -> dict[str, TransferredAnnotation]:
    """Merge the two transfer routes, recording where both agreed.

    Sequence takes precedence when both fire, because a bit score is directly comparable
    across queries in a way a TM-score is not. A protein reached by both routes has its
    route recorded as ``sequence+structure``, which is the strongest transferred evidence
    available here.
    """
    combined: dict[str, TransferredAnnotation] = dict(sequence_transfers)
    for accession, structural in structural_transfers.items():
        existing = combined.get(accession)
        if existing is None:
            combined[accession] = structural
            continue
        agreed = tuple(sorted(set(existing.terms) & set(structural.terms)))
        combined[accession] = TransferredAnnotation(
            accession=accession,
            # Both routes agreeing on a term is stronger than either alone; terms unique to
            # one route are kept, but the shared ones lead.
            terms=agreed + tuple(sorted(set(existing.terms) | set(structural.terms) - set(agreed))),
            donors=tuple(dict.fromkeys(existing.donors + structural.donors)),
            best_score=existing.best_score,
            best_evalue=min(existing.best_evalue, structural.best_evalue),
            n_donors_agreeing=existing.n_donors_agreeing + structural.n_donors_agreeing,
            route="sequence+structure",
        )
    return combined


def merge_observed_and_transferred(
    observed: Mapping[str, FunctionalRecord],
    transferred: Mapping[str, TransferredAnnotation],
) -> dict[str, tuple[str, ...]]:
    """Terms per accession, observed taking precedence over transferred.

    Provided for convenience only. Callers reporting a result must still distinguish the
    two sources; this exists for steps where the origin genuinely does not matter, such as
    counting how many proteins have any term at all.
    """
    merged: dict[str, tuple[str, ...]] = {}
    for accession, record in observed.items():
        if record.experimental_terms:
            merged[accession] = record.experimental_terms
    for accession, item in transferred.items():
        merged.setdefault(accession, item.terms)
    return merged


def donor_sequences_from(
    functions: Mapping[str, FunctionalRecord],
    sequences: Mapping[str, str],
) -> dict[str, str]:
    """Sequences for every protein eligible to donate annotation."""
    return {accession: sequences[accession] for accession in experimental_donors(functions, sequences)}
