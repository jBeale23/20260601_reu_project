"""Align domain-family sequences into a common frame.

The alignment itself lives in :mod:`domain_layout.msa` alongside the other external-tool
adapters; this module is the domain-family view of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout.msa import BACKEND_AUTO, align_sequences

if TYPE_CHECKING:
    from motif_conservation.families import DomainSlice


@dataclass(frozen=True, slots=True)
class AlignedFamily:
    """Aligned domain sequences for one family."""

    domain_family: str
    accessions: tuple[str, ...]
    aligned_sequences: tuple[str, ...]
    unaligned_sequences: tuple[str, ...]
    backend: str = ""


def align_domain_family(
    domain_family: str,
    slices: list[DomainSlice],
    *,
    backend: str = BACKEND_AUTO,
    threads: int = 1,
) -> AlignedFamily:
    """Align domain slices into a common frame.

    The backend that produced the alignment is carried on the result, so a run that fell
    back to the progressive aligner cannot be mistaken for a MAFFT run downstream.
    """
    accessions = [item.accession for item in slices]
    sequences = [item.sequence for item in slices]
    result = align_sequences(sequences, backend=backend, threads=threads)
    return AlignedFamily(
        domain_family=domain_family,
        accessions=tuple(accessions),
        aligned_sequences=result.aligned,
        unaligned_sequences=tuple(sequences),
        backend=result.backend,
    )
