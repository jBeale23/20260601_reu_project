"""Multiple-sequence alignment layer: MAFFT when installed, progressive fallback otherwise.

Structured domains are alignable, so they go to a real MSA rather than the alignment-free
comparison used for IDRs (:mod:`domain_layout.shark`). `MAFFT
<https://mafft.cbrc.jp/alignment/software/>`_ is the aligner of record here; it is a
compiled binary rather than a Python package, so it is invoked as a subprocess.

When ``mafft`` is not on ``PATH`` this module falls back to a progressive alignment built
on Biopython's pairwise aligner: each sequence is aligned in turn against the running
consensus. That is markedly worse than MAFFT - no iterative refinement, no guide tree, and
order-dependent - so the active backend is recorded on every alignment and reported in the
run summary. A fallback alignment is never presented as a MAFFT alignment.

Determinism
-----------
MAFFT is run with a fixed thread count (default 1). Its progressive stages are
thread-count-independent, but pinning the value keeps a rerun byte-identical without
having to reason about which algorithm ``--auto`` selected on a given machine.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from Bio.Align import PairwiseAligner

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

BACKEND_MAFFT = "mafft"
BACKEND_PROGRESSIVE = "progressive"
BACKEND_AUTO = "auto"

MAFFT_EXECUTABLE = "mafft"

# Generous, but bounded: a pathological family must not hold a cluster job open until it
# hits its wall-clock limit with nothing to show.
DEFAULT_MAFFT_TIMEOUT_SECONDS = 3600

# MAFFT's own guidance is that --auto degrades to progressive methods well past this, and
# alignment quality with it. Larger families are sub-sampled before alignment rather than
# aligned badly; see motif_conservation.analyze.
LARGE_FAMILY_WARNING = 5000


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Aligned sequences plus the backend that produced them."""

    aligned: tuple[str, ...]
    backend: str


def mafft_available() -> bool:
    """Whether the ``mafft`` binary can be found on ``PATH``."""
    return shutil.which(MAFFT_EXECUTABLE) is not None


def active_backend(*, backend: str = BACKEND_AUTO) -> str:
    """Report which alignment backend a run would use."""
    if backend == BACKEND_PROGRESSIVE:
        return BACKEND_PROGRESSIVE
    return BACKEND_MAFFT if mafft_available() else BACKEND_PROGRESSIVE


def _write_fasta(sequences: Sequence[str], path: Path) -> None:
    """Write sequences under synthetic IDs.

    Synthetic IDs rather than accessions: MAFFT truncates names at whitespace and a
    duplicate or malformed identifier would silently collapse two rows into one. Position
    is the only identity that matters here, since the caller pairs results back up by
    index.
    """
    lines: list[str] = []
    for index, sequence in enumerate(sequences):
        lines.append(f">s{index}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_fasta(text: str) -> dict[str, str]:
    """Parse aligned FASTA into an id-to-sequence map."""
    records: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0] if len(line) > 1 else ""
            records.setdefault(current, [])
        elif current is not None:
            records[current].append(line)
    return {name: "".join(parts) for name, parts in records.items()}


def _mafft_command(input_path: Path, *, threads: int) -> list[str]:
    """Build the MAFFT invocation.

    ``--auto`` picks the algorithm from the input size, which is what makes one command
    correct for both a six-member family and a five-thousand-member one. ``--anysymbol``
    is required rather than cosmetic: these sequences contain X, U, B and Z, and without
    it MAFFT rejects or rewrites them. ``--quiet`` keeps the progress report off stderr,
    where it would otherwise dominate a cluster log.
    """
    return [
        MAFFT_EXECUTABLE,
        "--auto",
        "--anysymbol",
        "--quiet",
        "--thread",
        str(max(1, threads)),
        str(input_path),
    ]


def _run_mafft(
    sequences: Sequence[str],
    *,
    threads: int,
    timeout: int,
) -> list[str] | None:
    """Align with MAFFT, or return None if it is unavailable or fails."""
    if not mafft_available():
        return None

    with tempfile.TemporaryDirectory(prefix="mafft_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.fasta"
        _write_fasta(sequences, input_path)
        command = _mafft_command(input_path, threads=threads)
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, path is ours
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "mafft timed out after %ss on %s sequence(s); falling back to the progressive aligner.",
                timeout,
                len(sequences),
            )
            return None
        except (subprocess.CalledProcessError, OSError):
            logger.warning(
                "mafft failed on %s sequence(s); falling back to the progressive aligner.",
                len(sequences),
                exc_info=True,
            )
            return None

    records = _parse_fasta(completed.stdout)
    try:
        # Re-keyed by the synthetic IDs rather than trusting output order. MAFFT preserves
        # input order without --reorder, but a silent reordering would misattribute every
        # residue to the wrong protein, so it is not worth assuming.
        ordered = [records[f"s{index}"] for index in range(len(sequences))]
    except KeyError:
        logger.warning(
            "mafft returned %s of %s expected records; falling back to the progressive aligner.",
            len(records),
            len(sequences),
        )
        return None

    # MAFFT lower-cases residues in several modes. Downstream charge and composition
    # analysis looks residues up in case-sensitive tables, so an unconverted alignment
    # would score as if every residue were unknown.
    return [sequence.upper() for sequence in ordered]


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
        counts = Counter(row[col] for row in msa if row[col] != "-")
        if not counts:
            chars.append("X")
            continue
        chars.append(counts.most_common(1)[0][0])
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


def progressive_msa(sequences: Sequence[str]) -> list[str]:
    """Align sequences progressively against a running consensus.

    The fallback for when MAFFT is not installed. It has no guide tree and no refinement
    pass, so the result depends on input order and degrades as families grow; it exists so
    the pipeline produces something reproducible on a machine without the binary, not
    because it is competitive with MAFFT.
    """
    if not sequences:
        return []
    if len(sequences) == 1:
        return list(sequences)

    aligner = _build_aligner()
    msa = [sequences[0]]
    for sequence in sequences[1:]:
        consensus = _msa_consensus(msa)
        alignments = aligner.align(consensus, sequence)
        best = next(iter(alignments))
        msa = _project_msa(msa, str(best[0]))
        msa.append(str(best[1]))
    return msa


def align_sequences(
    sequences: Sequence[str],
    *,
    backend: str = BACKEND_AUTO,
    threads: int = 1,
    timeout: int = DEFAULT_MAFFT_TIMEOUT_SECONDS,
) -> AlignmentResult:
    """Align a set of sequences, preferring MAFFT.

    Args:
        sequences: Unaligned sequences, in the order results should come back.
        backend: ``auto``, ``mafft``, or ``progressive``.
        threads: MAFFT thread count. Left at 1 inside worker processes, where the pool
            already owns every core.
        timeout: Seconds before MAFFT is abandoned for the fallback.

    Returns:
        The aligned sequences, padded to equal length, and the backend that produced them.
    """
    if not sequences:
        return AlignmentResult(aligned=(), backend=active_backend(backend=backend))
    if len(sequences) == 1:
        # Nothing to align. Reporting a backend here would credit MAFFT with work it never
        # did, so the trivial case is labelled as such.
        return AlignmentResult(aligned=tuple(sequences), backend=BACKEND_PROGRESSIVE)

    if len(sequences) > LARGE_FAMILY_WARNING:
        logger.warning(
            "aligning %s sequences in one family; alignment quality and runtime both degrade past ~%s.",
            len(sequences),
            LARGE_FAMILY_WARNING,
        )

    aligned: list[str] | None = None
    resolved = BACKEND_PROGRESSIVE
    if backend in {BACKEND_AUTO, BACKEND_MAFFT}:
        aligned = _run_mafft(sequences, threads=threads, timeout=timeout)
        if aligned is not None:
            resolved = BACKEND_MAFFT

    if aligned is None:
        aligned = progressive_msa(sequences)

    width = max((len(sequence) for sequence in aligned), default=0)
    return AlignmentResult(
        aligned=tuple(sequence.ljust(width, "-") for sequence in aligned),
        backend=resolved,
    )
