"""Layout-based JDP classification: class A/B/C, subclass, and novel-category detection.

Where the v1 classifier reads a single InterPro architecture string, this layer reads the
full domain complement, the disorder architecture, and the alignment-free similarity of a
protein's unalignable regions to curated reference JDPs. That combination is what makes a
"not A, not B, not the usual C" call defensible: a protein is flagged as a **novel-category
candidate** only when its J-domain context, its partner domains, and its IDR similarity all
fail to match the reference classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain_layout.constants import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_UNKNOWN,
    FAMILY_DNAJ_C,
    FAMILY_GF_RICH,
    FAMILY_J_DOMAIN,
    FAMILY_OTHER,
    FAMILY_ZINC_FINGER,
    GF_RICH_FRACTION_THRESHOLD,
    GF_RICH_WINDOW,
    HIGH_IDR_FRACTION,
    KIND_IDR,
    KIND_STRUCTURED_DOMAIN,
    NOVELTY_THRESHOLD,
    ROUTE_SHARK,
    SUBCLASS_A_CANONICAL,
    SUBCLASS_A_NO_CTD,
    SUBCLASS_B_CANONICAL,
    SUBCLASS_B_GF_ONLY,
    SUBCLASS_C_J_ONLY,
    SUBCLASS_C_MEMBRANE,
    SUBCLASS_C_MULTIDOMAIN,
    SUBCLASS_C_SECRETORY,
    SUBCLASS_C_UNASSIGNED,
    SUBCLASS_UNKNOWN,
)
from domain_layout.records import protein_dict_from_record
from jdp_classifier.hpd import has_hpd_motif
from jdp_classifier.localization import scan_entries_for_localization

if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain_layout.records import ProteinDomainRecord
    from domain_layout.regions import Region
    from domain_layout.shark import SharkMatch

# Novelty weights. They sum to 1.0 so the score is directly interpretable.
NOVELTY_WEIGHT_NO_HPD = 0.25
NOVELTY_WEIGHT_J_DOMAIN_PLACEMENT = 0.15
NOVELTY_WEIGHT_NO_CANONICAL_PARTNER = 0.20
NOVELTY_WEIGHT_UNKNOWN_PARTNER = 0.15
NOVELTY_WEIGHT_SHARK_DISSIMILAR = 0.25

# Below this SHARK similarity a protein's unalignable regions are considered unrelated
# to every reference class profile.
NOVEL_SHARK_SIMILARITY = 0.35

# Novelty scores are rounded to this many decimals before being compared and reported, so
# the flag always agrees with the number in the output table.
NOVELTY_SCORE_DECIMALS = 4

_GF_RESIDUES = frozenset("GF")
_J_DOMAIN_N_TERMINAL_LIMIT = 0.25


@dataclass(frozen=True, slots=True)
class ReferenceRegion:
    """One curated reference region used as a SHARK comparison target."""

    reference_id: str
    accession: str
    jdp_class: str
    kind: str
    sequence: str


@dataclass(frozen=True, slots=True)
class LayoutEvidence:
    """Structured and disordered evidence extracted from one protein layout."""

    accession: str
    protein_length: int
    has_j_domain: bool
    has_hpd: bool
    has_dnaj_c: bool
    has_zinc_finger_like: bool
    has_gf_rich_region: bool
    has_transmembrane: bool
    has_signal_peptide: bool
    j_domain_position: str
    n_structured_domains: int
    unknown_partner_families: int
    idr_fraction: float
    domain_family_layout: str
    structured_families: tuple[str, ...] = ()

    @property
    def has_canonical_partner(self) -> bool:
        """Whether a canonical class A/B partner module accompanies the J-domain."""
        return self.has_dnaj_c or self.has_zinc_finger_like or self.has_gf_rich_region


@dataclass(frozen=True, slots=True)
class LayoutClassification:
    """Class call for one protein, with subclass, confidence, and novelty."""

    predicted_class: str
    predicted_subclass: str
    class_confidence: str
    novelty_score: float
    novel_class_candidate: bool
    evidence_tags: tuple[str, ...] = field(default=())


def gf_fraction(sequence: str) -> float:
    """Fraction of glycine and phenylalanine residues in a sequence."""
    if not sequence:
        return 0.0
    return sum(1 for residue in sequence.upper() if residue in _GF_RESIDUES) / len(sequence)


def max_gf_window_fraction(sequence: str, *, window: int = GF_RICH_WINDOW) -> float:
    """Highest G/F fraction over any window of ``window`` residues.

    A G/F-rich block is a local compositional feature, typically 30-60 residues just
    after the J-domain. Scoring whole regions instead would make detection depend on
    where the disorder predictor happened to place region boundaries: the same block
    scores above threshold on its own but below it once merged into a longer IDR.
    Scanning windows keeps the class call identical across disorder backends.
    """
    if not sequence:
        return 0.0
    if len(sequence) <= window:
        return gf_fraction(sequence)

    upper = sequence.upper()
    hits = [1 if residue in _GF_RESIDUES else 0 for residue in upper]
    current = sum(hits[:window])
    best = current
    for index in range(window, len(hits)):
        current += hits[index] - hits[index - window]
        best = max(best, current)
    return best / window


def _j_domain_position(regions: Sequence[Region], protein_length: int) -> str:
    j_regions = [region for region in regions if region.family == FAMILY_J_DOMAIN]
    if not j_regions or protein_length <= 0:
        return "unknown"
    start = min(region.start for region in j_regions)
    end = max(region.end for region in j_regions)
    if start <= max(1, int(_J_DOMAIN_N_TERMINAL_LIMIT * protein_length)):
        return "n_terminal"
    if end >= protein_length - max(1, int(_J_DOMAIN_N_TERMINAL_LIMIT * protein_length)):
        return "c_terminal"
    return "internal"


def _has_gf_rich(regions: Sequence[Region]) -> bool:
    """Whether any annotated or compositionally G/F-rich unalignable region is present.

    Every unalignable region counts, not only the ones the disorder predictor called an
    IDR: a G/F-rich block often sits in a linker that scores below the disorder
    threshold, and it is still the class B signature.
    """
    for region in regions:
        if region.family == FAMILY_GF_RICH:
            return True
        if region.route == ROUTE_SHARK and max_gf_window_fraction(region.sequence) >= GF_RICH_FRACTION_THRESHOLD:
            return True
    return False


def region_idr_fraction(regions: Sequence[Region], protein_length: int) -> float:
    """Fraction of residues in IDR regions *after* annotated domains are carved out.

    This is more conservative than the raw predictor output: residues covered by an
    InterPro domain are never counted as disordered, which matters for charged helical
    domains (the J-domain in particular) that charge/hydropathy predictors over-call.
    """
    if protein_length <= 0:
        return 0.0
    covered = sum(region.length for region in regions if region.kind == KIND_IDR)
    return covered / protein_length


def build_evidence(record: ProteinDomainRecord, regions: Sequence[Region]) -> LayoutEvidence:
    """Collect classification evidence from a protein's routed regions."""
    families = [region.family for region in regions if region.kind == KIND_STRUCTURED_DOMAIN]
    protein_length = len(record.sequence) or record.length
    j_regions = [region for region in regions if region.family == FAMILY_J_DOMAIN]
    localization = scan_entries_for_localization(protein_dict_from_record(record))

    return LayoutEvidence(
        accession=record.accession,
        protein_length=protein_length,
        has_j_domain=bool(j_regions),
        has_hpd=any(has_hpd_motif(region.sequence) for region in j_regions),
        has_dnaj_c=FAMILY_DNAJ_C in families,
        has_zinc_finger_like=FAMILY_ZINC_FINGER in families,
        has_gf_rich_region=_has_gf_rich(regions),
        has_transmembrane=localization.has_transmembrane,
        has_signal_peptide=localization.has_signal_peptide,
        j_domain_position=_j_domain_position(regions, protein_length),
        n_structured_domains=len(families),
        unknown_partner_families=sum(1 for family in families if family == FAMILY_OTHER),
        idr_fraction=region_idr_fraction(regions, protein_length),
        domain_family_layout=">".join(families),
        structured_families=tuple(dict.fromkeys(families)),
    )


def novelty_score(evidence: LayoutEvidence, shark_match: SharkMatch | None) -> tuple[float, tuple[str, ...]]:
    """Score how poorly a J-domain protein matches the reference class templates.

    Returns:
        Tuple of the score in ``[0, 1]`` and the tags explaining which terms fired.
    """
    if not evidence.has_j_domain:
        return 0.0, ()

    score = 0.0
    tags: list[str] = []

    if not evidence.has_hpd:
        score += NOVELTY_WEIGHT_NO_HPD
        tags.append("no_hpd_in_j_domain")
    if evidence.j_domain_position not in {"n_terminal", "unknown"}:
        score += NOVELTY_WEIGHT_J_DOMAIN_PLACEMENT
        tags.append(f"j_domain_{evidence.j_domain_position}")
    if not evidence.has_canonical_partner:
        score += NOVELTY_WEIGHT_NO_CANONICAL_PARTNER
        tags.append("no_canonical_partner_domain")
    if evidence.unknown_partner_families > 0 and not evidence.has_canonical_partner:
        score += NOVELTY_WEIGHT_UNKNOWN_PARTNER
        tags.append("unannotated_partner_domain")
    if shark_match is None:
        tags.append("no_reference_similarity")
    elif shark_match.score < NOVEL_SHARK_SIMILARITY:
        score += NOVELTY_WEIGHT_SHARK_DISSIMILAR
        tags.append("shark_dissimilar_to_reference_classes")

    return min(1.0, score), tuple(tags)


def _class_c_subclass(evidence: LayoutEvidence) -> str:
    """Subdivide the class C catch-all by localization and domain layout."""
    if evidence.has_transmembrane:
        return SUBCLASS_C_MEMBRANE
    if evidence.has_signal_peptide:
        return SUBCLASS_C_SECRETORY
    if set(evidence.structured_families) == {FAMILY_J_DOMAIN}:
        return SUBCLASS_C_J_ONLY
    if evidence.n_structured_domains > 1:
        return SUBCLASS_C_MULTIDOMAIN
    return SUBCLASS_C_UNASSIGNED


def _class_and_subclass(evidence: LayoutEvidence) -> tuple[str, str]:
    """Assign class A/B/C and its subclass from the domain complement.

    Class A carries the zinc-finger-like cysteine-rich region (with or without the
    C-terminal substrate-binding domain); class B carries the C-terminal domain or a
    G/F-rich region but no zinc finger; everything else with a J-domain is class C.
    """
    if not evidence.has_j_domain:
        return CLASS_UNKNOWN, SUBCLASS_UNKNOWN
    if evidence.has_zinc_finger_like:
        return CLASS_A, SUBCLASS_A_CANONICAL if evidence.has_dnaj_c else SUBCLASS_A_NO_CTD
    if evidence.has_dnaj_c:
        return CLASS_B, SUBCLASS_B_CANONICAL
    if evidence.has_gf_rich_region and evidence.n_structured_domains >= 1:
        return CLASS_B, SUBCLASS_B_GF_ONLY
    return CLASS_C, _class_c_subclass(evidence)


def _confidence(evidence: LayoutEvidence, predicted_class: str) -> str:
    if predicted_class == CLASS_UNKNOWN:
        return "low"
    if evidence.has_hpd and evidence.has_canonical_partner:
        return "high"
    if evidence.has_hpd or evidence.has_canonical_partner:
        return "medium"
    return "low"


def classify_layout(
    evidence: LayoutEvidence,
    *,
    shark_match: SharkMatch | None = None,
) -> LayoutClassification:
    """Assign class, subclass, confidence, and novelty from layout evidence.

    Structure and novelty are orthogonal axes and are reported separately.
    ``predicted_subclass`` always describes the domain architecture that was found, even
    for a novel-category candidate: a class A protein whose J-domain lacks HPD is still
    architecturally ``a_canonical``, and overwriting that would discard the very evidence
    a reviewer needs. Novelty is carried by ``novelty_score``, ``novel_class_candidate``,
    and the ``novel_class_candidate`` evidence tag.
    """
    predicted_class, subclass = _class_and_subclass(evidence)
    raw_score, tags = novelty_score(evidence, shark_match)

    # Decide on the same rounded number that is reported, not the raw sum. Several weight
    # combinations land exactly on the threshold (0.15 + 0.20 + 0.15, and 0.25 + 0.25),
    # and none of the weights is exactly representable in binary floating point. Comparing
    # the raw sum would make "is this a novel candidate?" depend on which terms happened to
    # fire and in what order, so a future weight change could silently unflag a whole
    # category of proteins.
    score = round(raw_score, NOVELTY_SCORE_DECIMALS)
    is_novel = evidence.has_j_domain and score >= NOVELTY_THRESHOLD

    tag_list = list(tags)
    if evidence.has_transmembrane:
        tag_list.append("membrane_associated")
    if evidence.has_signal_peptide:
        tag_list.append("secretory_signal")
    if evidence.idr_fraction >= HIGH_IDR_FRACTION:
        tag_list.append("idr_dominated")
    if is_novel:
        tag_list.append("novel_class_candidate")

    return LayoutClassification(
        predicted_class=predicted_class,
        predicted_subclass=subclass,
        class_confidence=_confidence(evidence, predicted_class),
        novelty_score=score,
        novel_class_candidate=is_novel,
        evidence_tags=tuple(dict.fromkeys(tag_list)),
    )
