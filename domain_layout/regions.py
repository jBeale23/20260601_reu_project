"""Split a protein into residue regions and route each one to MSA or SHARK.

This is the "residue 1..x" step of the pipeline. InterPro domain matches define the
structured blocks; metapredict (or the FoldIndex fallback) defines which of the
remaining residues are disordered. Every residue ends up in exactly one region, and
every region carries a route:

* ``msa`` — structurally alignable domains (J-domain, DnaJ C-terminal, zinc-finger-like,
  other annotated domains). These feed the progressive-MSA / motif-conservation layer.
* ``shark`` — IDRs, G/F-rich low-complexity blocks, linkers, and termini. These feed the
  alignment-free SHARK layer.
* ``skip`` — fragments too short to score either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.constants import (
    ALIGNABLE_FAMILIES,
    DOMAIN_ENTRY_TYPES,
    FAMILY_OTHER,
    IDR_BLOCK_FRACTION,
    KIND_IDR,
    KIND_LINKER,
    KIND_STRUCTURED_DOMAIN,
    KIND_TERMINUS,
    MAX_INTERVAL_OVERLAP_FRACTION,
    MIN_DOMAIN_REGION_LENGTH,
    MIN_MSA_REGION_LENGTH,
    MIN_REGION_PIECE_LENGTH,
    MIN_SHARK_REGION_LENGTH,
    ROUTE_MSA,
    ROUTE_SHARK,
    ROUTE_SKIP,
    SIGNATURE_FAMILY_LABELS,
    WHOLE_PROTEIN_COVERAGE_FRACTION,
)
from domain_layout.records import MEMBER_DATABASE_PRIORITY
from motif_conservation.charge_alphabet import sequence_net_charge

if TYPE_CHECKING:
    from domain_layout.disorder import DisorderPrediction
    from domain_layout.records import DomainEntry, ProteinDomainRecord

_UNKNOWN_DATABASE_RANK = len(MEMBER_DATABASE_PRIORITY)


@dataclass(frozen=True, slots=True)
class DomainInterval:
    """One accepted, non-overlapping structured-domain interval (1-based, inclusive)."""

    start: int
    end: int
    family: str
    signature: str
    source_database: str

    @property
    def length(self) -> int:
        """Residues covered by this interval."""
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class Region:
    """One routed residue region of a protein (1-based, inclusive coordinates)."""

    accession: str
    index: int
    start: int
    end: int
    kind: str
    family: str
    signature: str
    route: str
    mean_disorder: float
    disorder_fraction: float
    sequence: str

    @property
    def length(self) -> int:
        """Residues covered by this region."""
        return self.end - self.start + 1

    @property
    def net_charge(self) -> int:
        """Formal net charge of the region sequence."""
        return sequence_net_charge(self.sequence)

    def fasta_id(self) -> str:
        """Stable FASTA identifier: ``<accession>|<start>-<end>|<kind>``."""
        return f"{self.accession}|{self.start}-{self.end}|{self.kind}"


def family_for_entry(entry: DomainEntry) -> str | None:
    """Map a signature match to a canonical family, or ``None`` when it is not a domain."""
    label = SIGNATURE_FAMILY_LABELS.get(entry.accession)
    if label is not None:
        return label
    if entry.integrated:
        label = SIGNATURE_FAMILY_LABELS.get(entry.integrated)
        if label is not None:
            return label
    if entry.entry_type in DOMAIN_ENTRY_TYPES:
        return FAMILY_OTHER
    return None


def _database_rank(source_database: str) -> int:
    try:
        return MEMBER_DATABASE_PRIORITY.index(source_database)
    except ValueError:
        return _UNKNOWN_DATABASE_RANK


def candidate_intervals(record: ProteinDomainRecord) -> list[DomainInterval]:
    """Return every domain fragment that could define a structured region."""
    length = record.length or len(record.sequence)
    candidates: list[DomainInterval] = []
    for entry in record.entries:
        family = family_for_entry(entry)
        if family is None:
            continue
        for fragment in entry.fragments:
            start = max(1, fragment.start)
            end = min(length, fragment.end) if length else fragment.end
            if end - start + 1 < MIN_DOMAIN_REGION_LENGTH:
                continue
            # A match spanning nearly the whole protein is a family-level assignment
            # rather than a domain boundary — but only for unrecognized signatures.
            # A curated JDP family covering everything means the protein really is just
            # that domain (short UniProt fragments are exactly one J-domain), and
            # discarding it would leave the protein with no J-domain at all.
            spans_whole_protein = bool(length) and (end - start + 1) >= WHOLE_PROTEIN_COVERAGE_FRACTION * length
            if spans_whole_protein and family == FAMILY_OTHER:
                continue
            candidates.append(
                DomainInterval(
                    start=start,
                    end=end,
                    family=family,
                    signature=entry.accession,
                    source_database=entry.source_database,
                ),
            )
    return candidates


def _longest_free_run(free: list[bool], start: int, end: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for position in range(start, end + 2):
        is_free = position <= end and free[position - 1]
        if is_free and run_start is None:
            run_start = position
        elif not is_free and run_start is not None:
            candidate = (run_start, position - 1)
            if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                best = candidate
            run_start = None
    return best


def select_domain_intervals(record: ProteinDomainRecord) -> list[DomainInterval]:
    """Resolve overlapping signature matches into disjoint structured-domain intervals.

    Candidates are ranked by whether they map to a known JDP family, then by member
    database reliability, then by length. Each accepted interval is trimmed to the
    residues still free; candidates that lose more than
    ``MAX_INTERVAL_OVERLAP_FRACTION`` of their length are dropped.
    """
    length = record.length or len(record.sequence)
    if length <= 0:
        return []

    candidates = sorted(
        candidate_intervals(record),
        key=lambda interval: (
            interval.family == FAMILY_OTHER,
            _database_rank(interval.source_database),
            -interval.length,
            interval.start,
        ),
    )

    free = [True] * length
    accepted: list[DomainInterval] = []
    for candidate in candidates:
        run = _longest_free_run(free, candidate.start, min(candidate.end, length))
        if run is None:
            continue
        run_length = run[1] - run[0] + 1
        if run_length < MIN_DOMAIN_REGION_LENGTH:
            continue
        if run_length < (1.0 - MAX_INTERVAL_OVERLAP_FRACTION) * candidate.length:
            continue
        for position in range(run[0], run[1] + 1):
            free[position - 1] = False
        accepted.append(
            DomainInterval(
                start=run[0],
                end=run[1],
                family=candidate.family,
                signature=candidate.signature,
                source_database=candidate.source_database,
            ),
        )

    return sorted(accepted, key=lambda interval: interval.start)


def _merge_short_pieces(pieces: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
    """Absorb pieces below the minimum length into an adjacent piece."""
    if not pieces:
        return []

    merged: list[list[int | bool]] = [list(pieces[0])]
    for start, end, is_idr in pieces[1:]:
        previous = merged[-1]
        previous_length = int(previous[1]) - int(previous[0]) + 1
        current_length = end - start + 1
        if current_length < MIN_REGION_PIECE_LENGTH or previous_length < MIN_REGION_PIECE_LENGTH:
            keep_idr = previous[2] if previous_length >= current_length else is_idr
            previous[1] = end
            previous[2] = keep_idr
            continue
        merged.append([start, end, is_idr])
    return [(int(start), int(end), bool(is_idr)) for start, end, is_idr in merged]


def _split_gap_by_disorder(start: int, end: int, disorder_flags: tuple[bool, ...]) -> list[tuple[int, int, bool]]:
    """Split an inter-domain block into alternating IDR / non-IDR pieces."""
    if end < start:
        return []

    pieces: list[tuple[int, int, bool]] = []
    piece_start = start
    current = bool(disorder_flags[start - 1]) if start - 1 < len(disorder_flags) else False
    for position in range(start + 1, end + 1):
        flag = bool(disorder_flags[position - 1]) if position - 1 < len(disorder_flags) else False
        if flag != current:
            pieces.append((piece_start, position - 1, current))
            piece_start = position
            current = flag
    pieces.append((piece_start, end, current))
    return _merge_short_pieces(pieces)


def _route_for(kind: str, family: str, length: int) -> str:
    if kind == KIND_STRUCTURED_DOMAIN:
        if family in ALIGNABLE_FAMILIES:
            return ROUTE_MSA if length >= MIN_MSA_REGION_LENGTH else ROUTE_SKIP
        return ROUTE_SHARK if length >= MIN_SHARK_REGION_LENGTH else ROUTE_SKIP
    return ROUTE_SHARK if length >= MIN_SHARK_REGION_LENGTH else ROUTE_SKIP


def _gap_kind(start: int, end: int, *, is_idr: bool, protein_length: int, disorder_fraction: float) -> str:
    if is_idr or disorder_fraction >= IDR_BLOCK_FRACTION:
        return KIND_IDR
    if start == 1 or end == protein_length:
        return KIND_TERMINUS
    return KIND_LINKER


def segment_protein(record: ProteinDomainRecord, disorder: DisorderPrediction) -> list[Region]:
    """Split one protein into routed regions covering every residue.

    Args:
        record: Fetched domain record (sequence plus signature matches).
        disorder: Per-residue disorder prediction for the same sequence.

    Returns:
        Regions ordered by start position; empty when the record has no sequence.
    """
    sequence = record.sequence
    length = len(sequence)
    if length == 0:
        return []

    intervals = [interval for interval in select_domain_intervals(record) if interval.start <= length]
    disorder_flags = disorder.residue_is_disordered()

    blocks: list[tuple[int, int, str, str, str]] = []
    cursor = 1
    for interval in intervals:
        end = min(interval.end, length)
        if interval.start > cursor:
            blocks.append((cursor, interval.start - 1, "", "", ""))
        blocks.append((interval.start, end, KIND_STRUCTURED_DOMAIN, interval.family, interval.signature))
        cursor = end + 1
    if cursor <= length:
        blocks.append((cursor, length, "", "", ""))

    regions: list[Region] = []
    for start, end, kind, family, signature in blocks:
        if kind == KIND_STRUCTURED_DOMAIN:
            spec = _RegionSpec(start=start, end=end, kind=kind, family=family, signature=signature)
            regions.append(_build_region(record, spec, disorder))
            continue
        for piece_start, piece_end, is_idr in _split_gap_by_disorder(start, end, disorder_flags):
            piece_kind = _gap_kind(
                piece_start,
                piece_end,
                is_idr=is_idr,
                protein_length=length,
                disorder_fraction=disorder.fraction_disordered_over(piece_start, piece_end),
            )
            regions.append(
                _build_region(record, _RegionSpec(start=piece_start, end=piece_end, kind=piece_kind), disorder),
            )

    ordered = sorted(regions, key=lambda region: region.start)
    return [
        Region(
            accession=region.accession,
            index=index,
            start=region.start,
            end=region.end,
            kind=region.kind,
            family=region.family,
            signature=region.signature,
            route=region.route,
            mean_disorder=region.mean_disorder,
            disorder_fraction=region.disorder_fraction,
            sequence=region.sequence,
        )
        for index, region in enumerate(ordered, start=1)
    ]


@dataclass(frozen=True, slots=True)
class _RegionSpec:
    """Internal span description before disorder statistics are attached."""

    start: int
    end: int
    kind: str
    family: str = ""
    signature: str = ""


def _build_region(record: ProteinDomainRecord, spec: _RegionSpec, disorder: DisorderPrediction) -> Region:
    length = spec.end - spec.start + 1
    return Region(
        accession=record.accession,
        index=0,
        start=spec.start,
        end=spec.end,
        kind=spec.kind,
        family=spec.family,
        signature=spec.signature,
        route=_route_for(spec.kind, spec.family, length),
        mean_disorder=disorder.mean_over(spec.start, spec.end),
        disorder_fraction=disorder.fraction_disordered_over(spec.start, spec.end),
        sequence=record.sequence[spec.start - 1 : spec.end],
    )


def regions_by_route(regions: list[Region], route: str) -> list[Region]:
    """Filter regions by route label."""
    return [region for region in regions if region.route == route]


def regions_by_family(regions: list[Region], family: str) -> list[Region]:
    """Filter regions by canonical domain family."""
    return [region for region in regions if region.family == family]
