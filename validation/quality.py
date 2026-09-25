"""Sequence-completeness quality control.

An incomplete gene model looks exactly like the dominant novelty signal — "a J-domain
with no partner domains" — so novel-category claims are only credible on sequences that
are plausibly complete. Every exclusion here is recorded with its reason so the effect of
the filter on any downstream count can be reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain_layout.records import ProteinDomainRecord

# A J-domain is ~70 residues. A JDP shorter than this cannot hold a J-domain plus any
# partner module, so "no partner domain" carries no information for such entries.
MIN_COMPLETE_LENGTH = 100

# When a single *domain* covers essentially the whole entry, the entry is a domain
# excerpt rather than a protein, whatever UniProt's fragment flag says.
DOMAIN_ONLY_COVERAGE = 0.95

# Only these entry types count towards the coverage rule. Family and
# homologous-superfamily signatures are *designed* to span an entire protein: NCBIfam,
# HAMAP, PANTHER and InterPro family entries all match a complete DnaJ end to end. Testing
# coverage against them excluded textbook full-length class A JDPs (J-domain + zinc finger
# + C-terminal domain) as though they were truncated, which is the opposite of the
# intended effect. A *domain* spanning the whole sequence is the real truncation signal.
COVERAGE_ENTRY_TYPES = frozenset({"domain"})

REASON_FRAGMENT_FLAG = "uniprot_fragment"
REASON_TOO_SHORT = "below_minimum_length"
REASON_DOMAIN_EXCERPT = "single_domain_covers_entry"
REASON_NO_SEQUENCE = "no_sequence"


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Whether one record is complete enough to support a novelty claim."""

    accession: str
    passes: bool
    reasons: tuple[str, ...]
    length: int

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "accession": self.accession,
            "passes": self.passes,
            "reasons": list(self.reasons),
            "length": self.length,
        }


def assess_record(record: ProteinDomainRecord) -> QualityAssessment:
    """Judge a single record's completeness against the documented criteria."""
    reasons: list[str] = []
    length = len(record.sequence) or record.length

    if not record.has_sequence:
        reasons.append(REASON_NO_SEQUENCE)
    if record.is_fragment:
        reasons.append(REASON_FRAGMENT_FLAG)
    if length and length < MIN_COMPLETE_LENGTH:
        reasons.append(REASON_TOO_SHORT)

    if length:
        for entry in record.entries:
            if entry.entry_type not in COVERAGE_ENTRY_TYPES:
                continue
            if entry.covered_length >= DOMAIN_ONLY_COVERAGE * length:
                reasons.append(REASON_DOMAIN_EXCERPT)
                break

    return QualityAssessment(
        accession=record.accession,
        passes=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        length=length,
    )


def assess_records(records: Iterable[ProteinDomainRecord]) -> dict[str, QualityAssessment]:
    """Assess completeness for every record, keyed by accession."""
    return {record.accession: assess_record(record) for record in records}


def summarize_quality(assessments: Iterable[QualityAssessment]) -> dict[str, object]:
    """Counts of passes and of each exclusion reason."""
    assessments = list(assessments)
    reason_counts: dict[str, int] = {}
    for assessment in assessments:
        for reason in assessment.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    passed = sum(1 for assessment in assessments if assessment.passes)
    return {
        "n_assessed": len(assessments),
        "n_passed": passed,
        "n_excluded": len(assessments) - passed,
        "exclusion_reasons": dict(sorted(reason_counts.items())),
    }
