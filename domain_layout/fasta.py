"""Write the per-class and per-route subFASTA files consumed by the MSA and SHARK layers."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from domain_layout import msa as msa_backend
from domain_layout.constants import KIND_STRUCTURED_DOMAIN, ROUTE_MSA, ROUTE_SHARK

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from domain_layout.regions import Region

logger = logging.getLogger(__name__)

FASTA_LINE_WIDTH = 60

# An alignment of one sequence is not an alignment.
MIN_SEQUENCES_TO_ALIGN = 2

# Ceiling on sequences aligned per family. MAFFT's own guidance is that alignment quality
# and runtime both degrade well before this on a single family, and a proteome-scale run
# would otherwise hand it six figures of J-domains at once.
DEFAULT_MAX_ALIGNED_SEQUENCES = 2000


def format_fasta(identifier: str, sequence: str, *, width: int = FASTA_LINE_WIDTH) -> str:
    """Render one wrapped FASTA record (including the trailing newline)."""
    lines = [sequence[index : index + width] for index in range(0, len(sequence), width)] or [""]
    body = "\n".join(lines)
    return f">{identifier}\n{body}\n"


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> int:
    """Write FASTA records to ``path`` and return how many were written."""
    entries = [(identifier, sequence) for identifier, sequence in records if sequence]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(format_fasta(identifier, sequence) for identifier, sequence in entries), encoding="utf-8")
    return len(entries)


def _sanitize(label: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)
    return cleaned or "unlabelled"


def write_subfastas(
    regions_by_accession: Mapping[str, Sequence[Region]],
    sequences_by_accession: Mapping[str, str],
    class_by_accession: Mapping[str, str],
    output_dir: Path,
) -> dict[str, int]:
    """Write every subFASTA the downstream layers need.

    Layout under ``output_dir``:

    * ``by_class/class_<X>.fasta`` — full-length sequences grouped by predicted class.
    * ``msa/<family>.fasta`` — structured domain regions, one file per domain family
      (MAFFT / progressive-MSA input).
    * ``shark/<kind>.fasta`` — unalignable regions grouped by region kind.
    * ``shark/class_<X>_<kind>.fasta`` — unalignable regions grouped by class and kind.

    Returns:
        Mapping of relative file path to record count.
    """
    by_class: dict[str, list[tuple[str, str]]] = {}
    msa_files: dict[str, list[tuple[str, str]]] = {}
    shark_files: dict[str, list[tuple[str, str]]] = {}

    for accession in sorted(regions_by_accession):
        jdp_class = class_by_accession.get(accession, "unclassified")
        sequence = sequences_by_accession.get(accession, "")
        if sequence:
            by_class.setdefault(_sanitize(jdp_class), []).append((accession, sequence))

        for region in regions_by_accession[accession]:
            if region.route == ROUTE_MSA:
                family = _sanitize(region.family or KIND_STRUCTURED_DOMAIN)
                msa_files.setdefault(family, []).append((region.fasta_id(), region.sequence))
            elif region.route == ROUTE_SHARK:
                kind = _sanitize(region.kind)
                shark_files.setdefault(kind, []).append((region.fasta_id(), region.sequence))
                shark_files.setdefault(f"class_{_sanitize(jdp_class)}_{kind}", []).append(
                    (region.fasta_id(), region.sequence),
                )

    counts: dict[str, int] = {}
    for jdp_class, records in sorted(by_class.items()):
        relative = f"by_class/class_{jdp_class}.fasta"
        counts[relative] = write_fasta(records, output_dir / relative)
    for family, records in sorted(msa_files.items()):
        relative = f"msa/{family}.fasta"
        counts[relative] = write_fasta(records, output_dir / relative)
    for name, records in sorted(shark_files.items()):
        relative = f"shark/{name}.fasta"
        counts[relative] = write_fasta(records, output_dir / relative)
    return counts


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file into (identifier, sequence) pairs, preserving file order."""
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(parts)))
            identifier = line[1:].split()[0] if len(line) > 1 else ""
            parts = []
        elif identifier is not None:
            parts.append(line)
    if identifier is not None:
        records.append((identifier, "".join(parts)))
    return records


def align_msa_subfastas(
    output_dir: Path,
    *,
    backend: str = msa_backend.BACKEND_AUTO,
    threads: int = 1,
    max_sequences: int = DEFAULT_MAX_ALIGNED_SEQUENCES,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    """Align every MSA-routed subFASTA and write the alignments alongside them.

    The subFASTAs under ``msa/`` are the structured half of the routing decision, and
    without this step they are only ever *input* to something else. This turns them into
    actual alignments under ``msa_aligned/``.

    Families are capped at ``max_sequences``. A proteome-scale run puts well over a
    hundred thousand J-domains in one family, which no aligner will handle in reasonable
    time or memory; past the cap a seeded random sample is aligned instead. Sampling is
    random rather than positional because the records are written in accession order, and
    UniProt accessions cluster by submitting project and organism.

    Returns:
        Per-family record of how many sequences were available, how many were aligned,
        the resulting column count, and the backend that produced it.
    """
    msa_dir = output_dir / "msa"
    if not msa_dir.is_dir():
        return {}

    aligned_dir = output_dir / "msa_aligned"
    report: dict[str, dict[str, object]] = {}

    for source in sorted(msa_dir.glob("*.fasta")):
        records = read_fasta(source)
        if len(records) < MIN_SEQUENCES_TO_ALIGN:
            continue

        available = len(records)
        if available > max_sequences:
            # Deterministic, seeded sampling for reproducibility; not a security context.
            rng = random.Random(seed)  # noqa: S311
            chosen = sorted(rng.sample(range(available), max_sequences))
            records = [records[index] for index in chosen]
            logger.info(
                "%s: sampling %s of %s sequences before alignment",
                source.name,
                max_sequences,
                available,
            )

        identifiers = [identifier for identifier, _sequence in records]
        result = msa_backend.align_sequences(
            [sequence for _identifier, sequence in records],
            backend=backend,
            threads=threads,
        )
        relative = f"msa_aligned/{source.stem}.aln.fasta"
        write_fasta(zip(identifiers, result.aligned, strict=True), aligned_dir / f"{source.stem}.aln.fasta")
        report[source.stem] = {
            "n_available": available,
            "n_aligned": len(records),
            "n_columns": len(result.aligned[0]) if result.aligned else 0,
            "backend": result.backend,
            "path": relative,
        }
    return report
