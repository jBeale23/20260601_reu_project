"""Gold-standard class labels derived from curated UniProt protein nomenclature.

The classifier assigns class A/B/C from domain architecture. To measure whether that is
right, labels must come from a source the classifier never sees. UniProt curators name
reviewed JDPs with an explicit subfamily, for example "DnaJ homolog subfamily B member
1", and that nomenclature *is* the DNAJA/DNAJB/DNAJC assignment used in the literature.

Only reviewed (Swiss-Prot) entries are used: unreviewed names are produced by automatic
annotation pipelines, which are themselves derived from domain composition and would make
the evaluation circular.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain_layout.records import ProteinDomainRecord

# "DnaJ homolog subfamily A member 1", "dnaJ homolog subfamily C member 7", and the
# hyphenated/parenthesised variants curators use.
SUBFAMILY_PATTERN = re.compile(r"dnaj\s+homolog\s+subfamily\s+([abc])\b", re.IGNORECASE)

# Curated entries only; automatic annotation is derived from domain content and would
# make this evaluation circular.
REVIEWED_SOURCE = "reviewed"


@dataclass(frozen=True, slots=True)
class GoldLabel:
    """One curated class assignment and the evidence it came from."""

    accession: str
    label: str
    source_name: str
    organism_name: str


def label_from_name(name: str) -> str | None:
    """Extract an A/B/C subfamily label from a curated protein name.

    Returns ``None`` when the name does not carry explicit subfamily nomenclature; those
    proteins are simply not part of the evaluation set rather than being guessed at.
    """
    match = SUBFAMILY_PATTERN.search(name or "")
    if match is None:
        return None
    return match.group(1).upper()


def gold_label(record: ProteinDomainRecord) -> GoldLabel | None:
    """Return the curated label for a record, or ``None`` when it is not labelled."""
    if record.source_database != REVIEWED_SOURCE:
        return None
    label = label_from_name(record.name)
    if label is None:
        return None
    return GoldLabel(
        accession=record.accession,
        label=label,
        source_name=record.name,
        organism_name=record.organism_name,
    )


def collect_gold_labels(records: Iterable[ProteinDomainRecord]) -> dict[str, GoldLabel]:
    """Build the labelled evaluation set from a domain store."""
    labels: dict[str, GoldLabel] = {}
    for record in records:
        label = gold_label(record)
        if label is not None:
            labels[record.accession] = label
    return labels
