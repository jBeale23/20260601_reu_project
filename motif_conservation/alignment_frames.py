"""Align domain-family sequences into a common frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from Bio.Align import PairwiseAligner

if TYPE_CHECKING:
    from motif_conservation.families import DomainSlice


@dataclass(frozen=True, slots=True)
class AlignedFamily:
    """Aligned domain sequences for one family."""

    domain_family: str
    accessions: tuple[str, ...]
    aligned_sequences: tuple[str, ...]
    unaligned_sequences: tuple[str, ...]


def _build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def _msa_consensus(msa: list[str]) -> str:
    if not msa:
        return ""
    length = len(msa[0])
    chars: list[str] = []
    for col in range(length):
        counts: dict[str, int] = {}
        for row in msa:
            aa = row[col]
            if aa == "-":
                continue
            counts[aa] = counts.get(aa, 0) + 1
        if not counts:
            chars.append("X")
            continue
        chars.append(max(counts, key=counts.get))
    return "".join(chars)


def _project_msa(msa: list[str], aligned_consensus: str) -> list[str]:
    """Insert gaps into each MSA row to match aligned consensus columns."""
    mapping: list[int | None] = []
    old_i = 0
    for aa in aligned_consensus:
        if aa == "-":
            mapping.append(None)
        else:
            mapping.append(old_i)
            old_i += 1

    projected: list[str] = []
    for row in msa:
        chars: list[str] = []
        for old_col in mapping:
            if old_col is None:
                chars.append("-")
            else:
                chars.append(row[old_col] if old_col < len(row) else "-")
        projected.append("".join(chars))
    return projected


def _pairwise_progressive_msa(sequences: list[str]) -> list[str]:
    """Build a simple progressive MSA without external aligner binaries."""
    if not sequences:
        return []
    if len(sequences) == 1:
        return list(sequences)

    aligner = _build_aligner()
    msa = [sequences[0]]
    for seq in sequences[1:]:
        consensus = _msa_consensus(msa)
        alignments = aligner.align(consensus, seq)
        best = next(iter(alignments))
        aligned_a = str(best[0])
        aligned_b = str(best[1])
        msa = _project_msa(msa, aligned_a)
        msa.append(aligned_b)
    return msa


def align_domain_family(domain_family: str, slices: list[DomainSlice]) -> AlignedFamily:
    """Align domain slices into a common frame."""
    accessions = [item.accession for item in slices]
    sequences = [item.sequence for item in slices]
    aligned = _pairwise_progressive_msa(sequences)
    max_len = max((len(seq) for seq in aligned), default=0)
    aligned = [seq.ljust(max_len, "-") for seq in aligned]
    return AlignedFamily(
        domain_family=domain_family,
        accessions=tuple(accessions),
        aligned_sequences=tuple(aligned),
        unaligned_sequences=tuple(sequences),
    )
