"""Map reference pocket residues onto target structures via sequence alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from Bio.Align import PairwiseAligner
from scipy.spatial import cKDTree

from structure_analysis.constants import AA3_TO_1

if TYPE_CHECKING:
    from structure_analysis.pdb_io import ResidueRecord

MappingMode = Literal["sbd_local", "full_length"]

MIN_SBD_ALIGNMENT_COVERAGE = 0.50


@dataclass(frozen=True, slots=True)
class PocketAlignmentContext:
    """Inputs for mapping reference pocket residues onto a target alignment."""

    reference_sequence: str
    target_sequence: str
    reference_resnums: list[int]
    reference_residues: list[ResidueRecord]
    sbd_residue_range: tuple[int, int]
    ref_pos_to_tgt_index: dict[int, int]
    aligned_pairs: list[tuple[int, str, str]]
    contact_ref_positions: set[int]
    mapping_mode: MappingMode
    sbd_alignment_coverage: float


@dataclass(frozen=True, slots=True)
class PocketMappingInputs:
    """Sequence and structure inputs for pocket mapping."""

    reference_sequence: str
    target_sequence: str
    reference_resnums: list[int]
    reference_residues: list[ResidueRecord]
    target_residues: list[ResidueRecord]
    sbd_residue_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SbdExtraction:
    """Reference SBD subsequence extracted from structure residue numbers."""

    sequence: str
    ref_index_to_sbd_pos: dict[int, int]
    sbd_pos_to_ref_index: dict[int, int]
    ref_resnum_to_sbd_pos: dict[int, int]


@dataclass(frozen=True, slots=True)
class TargetSbdExtraction:
    """Target SBD subsequence extracted from mapped full-structure indices."""

    sequence: str
    sbd_pos_to_target_index: dict[int, int]


@dataclass(frozen=True, slots=True)
class AlignmentMapping:
    """Result of mapping reference pocket positions onto a target sequence."""

    resnum_to_target_index: dict[int, int]
    warnings: list[str]
    contact_sequence_identity: float
    n_contact_mismatches: int
    sbd_target_indices: set[int]
    mapping_mode: MappingMode
    sbd_alignment_coverage: float
    ref_index_to_target_index: dict[int, int]


def _build_global_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def extract_sbd_sequence(
    residues: list[ResidueRecord],
    sbd_residue_range: tuple[int, int],
) -> SbdExtraction:
    """Extract SBD one-letter sequence and index maps from structure residues."""
    start, end = sbd_residue_range
    sbd_residues = sorted(
        (r for r in residues if start <= r.resnum <= end),
        key=lambda r: r.resnum,
    )
    ref_index_to_sbd_pos: dict[int, int] = {}
    sbd_pos_to_ref_index: dict[int, int] = {}
    ref_resnum_to_sbd_pos: dict[int, int] = {}
    chars: list[str] = []
    for sbd_pos, residue in enumerate(sbd_residues, start=1):
        one = AA3_TO_1.get(residue.resname, "X")
        chars.append(one)
        ref_index_to_sbd_pos[residue.index] = sbd_pos
        sbd_pos_to_ref_index[sbd_pos] = residue.index
        ref_resnum_to_sbd_pos[residue.resnum] = sbd_pos
    return SbdExtraction(
        sequence="".join(chars),
        ref_index_to_sbd_pos=ref_index_to_sbd_pos,
        sbd_pos_to_ref_index=sbd_pos_to_ref_index,
        ref_resnum_to_sbd_pos=ref_resnum_to_sbd_pos,
    )


def extract_target_sbd_sequence(
    target_residues: list[ResidueRecord],
    target_sbd_indices: set[int],
) -> TargetSbdExtraction:
    """Build a target SBD subsequence from mapped 1-based structure indices."""
    ordered = sorted(
        (r for r in target_residues if r.index in target_sbd_indices),
        key=lambda r: r.index,
    )
    sbd_pos_to_target_index: dict[int, int] = {}
    chars: list[str] = []
    for sbd_pos, residue in enumerate(ordered, start=1):
        chars.append(AA3_TO_1.get(residue.resname, "X"))
        sbd_pos_to_target_index[sbd_pos] = residue.index
    return TargetSbdExtraction(
        sequence="".join(chars),
        sbd_pos_to_target_index=sbd_pos_to_target_index,
    )


def _alignment_position_maps(
    ref_aln: str,
    tgt_aln: str,
) -> tuple[dict[int, int], list[tuple[int, str, str]]]:
    """Build ref_pos->tgt_pos map and list of (ref_pos, ref_char, tgt_char) pairs."""
    ref_pos_to_tgt_index: dict[int, int] = {}
    aligned_pairs: list[tuple[int, str, str]] = []
    ref_pos = 0
    tgt_pos = 0
    for r_char, t_char in zip(ref_aln, tgt_aln, strict=True):
        if r_char != "-":
            ref_pos += 1
        if t_char != "-":
            tgt_pos += 1
        if r_char != "-" and t_char != "-":
            ref_pos_to_tgt_index[ref_pos] = tgt_pos
            aligned_pairs.append((ref_pos, r_char, t_char))
    return ref_pos_to_tgt_index, aligned_pairs


def _contact_sequence_identity(
    aligned_pairs: list[tuple[int, str, str]],
    contact_ref_positions: set[int],
) -> float:
    """Fraction of identical aligned pairs at contact reference positions."""
    if not contact_ref_positions:
        return 0.0
    matches = 0
    total = 0
    for ref_pos, r_char, t_char in aligned_pairs:
        if ref_pos in contact_ref_positions:
            total += 1
            if r_char == t_char:
                matches += 1
    if total == 0:
        return 0.0
    return matches / total


def _count_contact_mismatches(
    mapping: dict[int, int],
    reference_resnums: list[int],
    reference_sequence: str,
    target_sequence: str,
    ref_resnum_to_seq_index: dict[int, int],
) -> int:
    n_mismatches = 0
    for resnum in reference_resnums:
        ref_seq_index = ref_resnum_to_seq_index.get(resnum)
        tgt_index = mapping.get(resnum)
        if ref_seq_index is None or tgt_index is None:
            continue
        if reference_sequence[ref_seq_index - 1] != target_sequence[tgt_index - 1]:
            n_mismatches += 1
    return n_mismatches


def map_sbd_window(
    reference_residues: list[ResidueRecord],
    ref_pos_to_tgt_index: dict[int, int],
    sbd_start_resnum: int,
    sbd_end_resnum: int,
) -> set[int]:
    """Map reference SBD UniProt residue range to target 1-based indices."""
    ref_resnum_to_seq_index = {r.resnum: r.index for r in reference_residues}
    sbd_indices: set[int] = set()
    for resnum in range(sbd_start_resnum, sbd_end_resnum + 1):
        ref_seq_index = ref_resnum_to_seq_index.get(resnum)
        if ref_seq_index is None:
            continue
        tgt_index = ref_pos_to_tgt_index.get(ref_seq_index)
        if tgt_index is not None:
            sbd_indices.add(tgt_index)
    return sbd_indices


def _map_pocket_from_alignment(context: PocketAlignmentContext) -> AlignmentMapping:
    warnings: list[str] = []
    ref_resnum_to_seq_index = {r.resnum: r.index for r in context.reference_residues}
    contact_sequence_identity = _contact_sequence_identity(
        context.aligned_pairs,
        context.contact_ref_positions,
    )

    mapping: dict[int, int] = {}
    for resnum in context.reference_resnums:
        ref_seq_index = ref_resnum_to_seq_index.get(resnum)
        if ref_seq_index is None:
            warnings.append(f"Reference residue {resnum} not found in reference structure.")
            continue
        tgt_index = context.ref_pos_to_tgt_index.get(ref_seq_index)
        if tgt_index is None:
            warnings.append(f"Reference residue {resnum} did not align to target sequence.")
            continue
        mapping[resnum] = tgt_index

    if not mapping:
        warnings.append("No reference pocket residues mapped to target.")

    n_contact_mismatches = _count_contact_mismatches(
        mapping,
        context.reference_resnums,
        context.reference_sequence,
        context.target_sequence,
        ref_resnum_to_seq_index,
    )

    sbd_target_indices = map_sbd_window(
        context.reference_residues,
        context.ref_pos_to_tgt_index,
        context.sbd_residue_range[0],
        context.sbd_residue_range[1],
    )
    if not sbd_target_indices:
        warnings.append("SBD window did not map to target sequence.")

    return AlignmentMapping(
        resnum_to_target_index=mapping,
        warnings=warnings,
        contact_sequence_identity=contact_sequence_identity,
        n_contact_mismatches=n_contact_mismatches,
        sbd_target_indices=sbd_target_indices,
        mapping_mode=context.mapping_mode,
        sbd_alignment_coverage=context.sbd_alignment_coverage,
        ref_index_to_target_index=dict(context.ref_pos_to_tgt_index),
    )


def _rough_full_length_ref_to_tgt(
    reference_sequence: str,
    target_sequence: str,
) -> dict[int, int]:
    """Global full-length alignment map from reference to target sequence indices."""
    aligner = _build_global_aligner()
    alignment = aligner.align(reference_sequence, target_sequence)[0]
    ref_aln = str(alignment[0])
    tgt_aln = str(alignment[1])
    ref_pos_to_tgt_index, _ = _alignment_position_maps(ref_aln, tgt_aln)
    return ref_pos_to_tgt_index


@dataclass(frozen=True, slots=True)
class SbdAlignmentContext:
    """Inputs for local SBD-domain alignment and pocket mapping."""

    ref_sbd: SbdExtraction
    target_sbd: TargetSbdExtraction
    reference_residues: list[ResidueRecord]
    reference_resnums: list[int]
    reference_sequence: str
    target_sequence: str
    sbd_residue_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FullLengthAlignmentContext:
    """Inputs for full-length sequence alignment pocket mapping."""

    reference_sequence: str
    target_sequence: str
    reference_resnums: list[int]
    reference_residues: list[ResidueRecord]
    sbd_residue_range: tuple[int, int]


def _align_sbd_domains(context: SbdAlignmentContext) -> AlignmentMapping:
    """Globally align reference and target SBD subsequences after rough window mapping."""
    aligner = _build_global_aligner()
    alignment = aligner.align(context.ref_sbd.sequence, context.target_sbd.sequence)[0]
    ref_aln = str(alignment[0])
    tgt_aln = str(alignment[1])
    ref_sbd_pos_to_tgt_sbd_pos, aligned_pairs = _alignment_position_maps(ref_aln, tgt_aln)

    ref_pos_to_tgt_index: dict[int, int] = {}
    for ref_sbd_pos, tgt_sbd_pos in ref_sbd_pos_to_tgt_sbd_pos.items():
        ref_index = context.ref_sbd.sbd_pos_to_ref_index.get(ref_sbd_pos)
        tgt_index = context.target_sbd.sbd_pos_to_target_index.get(tgt_sbd_pos)
        if ref_index is not None and tgt_index is not None:
            ref_pos_to_tgt_index[ref_index] = tgt_index

    contact_ref_positions = {
        context.ref_sbd.ref_resnum_to_sbd_pos[r]
        for r in context.reference_resnums
        if r in context.ref_sbd.ref_resnum_to_sbd_pos
    }
    n_sbd_mapped = len(ref_sbd_pos_to_tgt_sbd_pos)
    n_sbd_total = len(context.ref_sbd.sequence)
    sbd_coverage = n_sbd_mapped / n_sbd_total if n_sbd_total else 0.0

    sbd_target_indices = {
        context.target_sbd.sbd_pos_to_target_index[pos] for pos in context.target_sbd.sbd_pos_to_target_index
    }

    mapping = _map_pocket_from_alignment(
        PocketAlignmentContext(
            reference_sequence=context.reference_sequence,
            target_sequence=context.target_sequence,
            reference_resnums=context.reference_resnums,
            reference_residues=context.reference_residues,
            sbd_residue_range=context.sbd_residue_range,
            ref_pos_to_tgt_index=ref_pos_to_tgt_index,
            aligned_pairs=aligned_pairs,
            contact_ref_positions=contact_ref_positions,
            mapping_mode="sbd_local",
            sbd_alignment_coverage=round(sbd_coverage, 4),
        ),
    )
    return AlignmentMapping(
        resnum_to_target_index=mapping.resnum_to_target_index,
        warnings=mapping.warnings,
        contact_sequence_identity=mapping.contact_sequence_identity,
        n_contact_mismatches=mapping.n_contact_mismatches,
        sbd_target_indices=sbd_target_indices,
        mapping_mode="sbd_local",
        sbd_alignment_coverage=round(sbd_coverage, 4),
        ref_index_to_target_index=dict(ref_pos_to_tgt_index),
    )


def _align_full_length(context: FullLengthAlignmentContext) -> AlignmentMapping:
    """Global full-length alignment (v2 fallback)."""
    ref_resnum_to_seq_index = {r.resnum: r.index for r in context.reference_residues}
    contact_ref_indices = {
        ref_resnum_to_seq_index[r] for r in context.reference_resnums if r in ref_resnum_to_seq_index
    }

    aligner = _build_global_aligner()
    alignment = aligner.align(context.reference_sequence, context.target_sequence)[0]
    ref_aln = str(alignment[0])
    tgt_aln = str(alignment[1])
    ref_pos_to_tgt_index, aligned_pairs = _alignment_position_maps(ref_aln, tgt_aln)

    return _map_pocket_from_alignment(
        PocketAlignmentContext(
            reference_sequence=context.reference_sequence,
            target_sequence=context.target_sequence,
            reference_resnums=context.reference_resnums,
            reference_residues=context.reference_residues,
            sbd_residue_range=context.sbd_residue_range,
            ref_pos_to_tgt_index=ref_pos_to_tgt_index,
            aligned_pairs=aligned_pairs,
            contact_ref_positions=contact_ref_indices,
            mapping_mode="full_length",
            sbd_alignment_coverage=0.0,
        ),
    )


def _mapping_score(mapping: AlignmentMapping) -> tuple[int, float, int]:
    """Sort key: more contacts mapped, higher identity, fewer mismatches."""
    n_mapped = len(mapping.resnum_to_target_index)
    return (n_mapped, mapping.contact_sequence_identity, -mapping.n_contact_mismatches)


def map_reference_pocket(inputs: PocketMappingInputs) -> AlignmentMapping:
    """Map reference pocket residue numbers and SBD window onto target indices."""
    ref_sbd = extract_sbd_sequence(inputs.reference_residues, inputs.sbd_residue_range)
    n_expected = len(inputs.reference_resnums)
    full_length_context = FullLengthAlignmentContext(
        reference_sequence=inputs.reference_sequence,
        target_sequence=inputs.target_sequence,
        reference_resnums=inputs.reference_resnums,
        reference_residues=inputs.reference_residues,
        sbd_residue_range=inputs.sbd_residue_range,
    )

    if ref_sbd.sequence:
        rough_map = _rough_full_length_ref_to_tgt(inputs.reference_sequence, inputs.target_sequence)
        rough_sbd_indices = map_sbd_window(
            inputs.reference_residues,
            rough_map,
            inputs.sbd_residue_range[0],
            inputs.sbd_residue_range[1],
        )
        target_sbd = extract_target_sbd_sequence(inputs.target_residues, rough_sbd_indices)
        if target_sbd.sequence:
            sbd_mapping = _align_sbd_domains(
                SbdAlignmentContext(
                    ref_sbd=ref_sbd,
                    target_sbd=target_sbd,
                    reference_residues=inputs.reference_residues,
                    reference_resnums=inputs.reference_resnums,
                    reference_sequence=inputs.reference_sequence,
                    target_sequence=inputs.target_sequence,
                    sbd_residue_range=inputs.sbd_residue_range,
                ),
            )
            if sbd_mapping.sbd_alignment_coverage >= MIN_SBD_ALIGNMENT_COVERAGE and len(
                sbd_mapping.resnum_to_target_index
            ) >= max(1, n_expected // 2):
                full_mapping = _align_full_length(full_length_context)
                if _mapping_score(full_mapping) > _mapping_score(sbd_mapping):
                    return full_mapping
                return sbd_mapping

    return _align_full_length(full_length_context)


def expand_shell_residue_indices(
    residues: list[ResidueRecord],
    seed_indices: set[int],
    radius_angstrom: float,
    allowed_indices: set[int] | None = None,
) -> set[int]:
    """Return seed indices plus residues within radius of seed CA atoms."""
    by_index = {r.index: r for r in residues}
    seeds = [by_index[i] for i in seed_indices if i in by_index and by_index[i].ca_coord is not None]
    if not seeds:
        return set()

    expanded = set(seed_indices)
    candidate_indices = allowed_indices if allowed_indices is not None else set(by_index)
    candidates = [
        by_index[i]
        for i in candidate_indices
        if i in by_index and by_index[i].ca_coord is not None and i not in expanded
    ]
    if not candidates:
        return expanded

    seed_coords = np.array([s.ca_coord for s in seeds], dtype=float)
    cand_coords = np.array([c.ca_coord for c in candidates], dtype=float)
    tree = cKDTree(seed_coords)
    neighbors = tree.query_ball_point(cand_coords, radius_angstrom)
    for candidate, close_seed_indices in zip(candidates, neighbors, strict=True):
        if close_seed_indices:
            expanded.add(candidate.index)

    return expanded
