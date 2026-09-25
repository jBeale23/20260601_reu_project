"""Write the per-class and per-route subFASTA files consumed by the MSA and SHARK layers."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout import msa as msa_backend
from domain_layout.constants import KIND_STRUCTURED_DOMAIN, ROUTE_MSA, ROUTE_SHARK
from domain_layout.progress import progress

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

# Largest length ratio allowed inside one alignment group.
#
# This is arithmetic, not taste. Align a sequence of length L against one of length kL and
# the shorter contributes at most L of the kL columns, so the alignment is at least
# (1 - 1/k) gaps before any biology is considered. Bounding the ratio bounds the floor on
# the gap fraction: at 2.0 no group can be forced past 50% gaps by length alone.
#
# Measured need: the dnaj_c family is strongly bimodal - 46,245 regions near 27 residues
# and 41,802 near 125 - because a domain interrupted by an inserted zinc finger is stored
# as separate fragments alongside uninterrupted copies of the same domain. Aligning a
# 27-residue fragment against a 125-residue complete domain is a category error, and doing
# it across the whole family gave 92% gaps. Split by length band, the same regions align at
# 58% and 50%.
MAX_LENGTH_RATIO_IN_GROUP = 2.0


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


@dataclass(frozen=True, slots=True)
class AlignmentOptions:
    """How the MSA-routed subFASTAs are prepared and aligned."""

    backend: str = msa_backend.BACKEND_AUTO
    threads: int = 1
    max_sequences: int = DEFAULT_MAX_ALIGNED_SEQUENCES
    seed: int = 0
    # Split a family into length bands before aligning. Off only to reproduce the
    # unbanded behaviour for comparison.
    band_by_length: bool = True


DEFAULT_ALIGNMENT_OPTIONS = AlignmentOptions()


def length_bands(
    records: Sequence[tuple[str, str]],
    *,
    max_ratio: float = MAX_LENGTH_RATIO_IN_GROUP,
    min_members: int = MIN_SEQUENCES_TO_ALIGN,
) -> list[list[tuple[str, str]]]:
    """Partition records into groups whose members are within ``max_ratio`` in length.

    Greedy over sorted lengths: a band grows until admitting the next record would make it
    span more than ``max_ratio``, then a new band starts. That is the smallest number of
    bands satisfying the constraint, so nothing is split more finely than the arithmetic
    requires.

    Nothing is discarded. Every record lands in exactly one band, and bands too small to
    align are reported rather than dropped - a family that shatters into singletons is a
    finding about that family, not a reason to hide it.
    """
    if not records:
        return []

    ordered = sorted(records, key=lambda item: len(item[1]))
    bands: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for record in ordered:
        length = len(record[1])
        if current and length > max_ratio * len(current[0][1]):
            bands.append(current)
            current = []
        current.append(record)
    if current:
        bands.append(current)

    # Merge a trailing band that is too small back into its neighbour when doing so keeps
    # the ratio bound; otherwise it stays separate and is reported as unalignable.
    merged: list[list[tuple[str, str]]] = []
    for band in bands:
        if merged and len(band) < min_members and len(band[-1][1]) <= max_ratio * len(merged[-1][0][1]):
            merged[-1].extend(band)
        else:
            merged.append(band)
    return merged


def align_msa_subfastas(
    output_dir: Path,
    *,
    options: AlignmentOptions = DEFAULT_ALIGNMENT_OPTIONS,
) -> dict[str, dict[str, object]]:
    """Align every MSA-routed subFASTA and write the alignments alongside them.

    The subFASTAs under ``msa/`` are the structured half of the routing decision, and
    without this step they are only ever *input* to something else. This turns them into
    actual alignments under ``msa_aligned/``.

    Two things happen before an aligner sees a family, and neither discards data.

    First the family is split into length bands (:func:`length_bands`), because a family
    is not always one thing. A domain interrupted by an inserted zinc finger is stored as
    separate fragments beside uninterrupted copies of the same domain, so ``dnaj_c``
    contains a population near 27 residues and another near 125. Aligned together they
    gave 92% gaps; aligned as bands, 58% and 50%.

    Then each band is capped at ``max_sequences``, since a proteome-scale run puts well
    over a hundred thousand J-domains in one family and no aligner handles that. The cap
    is a seeded random sample rather than the first N, because records are written in
    accession order and UniProt accessions cluster by submitting project and organism.

    Returns:
        One entry per aligned band, keyed ``family`` or ``family__<low>-<high>`` when a
        family split, each carrying its own column count and gap fraction so the quality
        of every alignment is visible rather than assumed.
    """
    backend, threads = options.backend, options.threads
    max_sequences, seed = options.max_sequences, options.seed

    msa_dir = output_dir / "msa"
    if not msa_dir.is_dir():
        return {}

    aligned_dir = output_dir / "msa_aligned"
    report: dict[str, dict[str, object]] = {}

    for source in progress(sorted(msa_dir.glob("*.fasta")), description="Aligning families", unit="family"):
        records = read_fasta(source)
        if len(records) < MIN_SEQUENCES_TO_ALIGN:
            continue

        bands = length_bands(records) if options.band_by_length else [list(records)]
        for band in bands:
            lengths = [len(sequence) for _identifier, sequence in band]
            low, high = min(lengths), max(lengths)
            key = source.stem if len(bands) == 1 else f"{source.stem}__{low}-{high}"

            if len(band) < MIN_SEQUENCES_TO_ALIGN:
                # Reported, not dropped: a band of one is a fact about the family.
                report[key] = {
                    "n_available": len(band),
                    "n_aligned": 0,
                    "length_range": [low, high],
                    "note": "too few members in this length band to align",
                }
                continue

            available = len(band)
            selected = band
            if available > max_sequences:
                # Deterministic, seeded sampling for reproducibility; not a security context.
                rng = random.Random(seed)  # noqa: S311
                chosen = sorted(rng.sample(range(available), max_sequences))
                selected = [band[index] for index in chosen]
                logger.info("%s: sampling %s of %s sequences", key, max_sequences, available)

            result = msa_backend.align_sequences(
                [sequence for _identifier, sequence in selected],
                backend=backend,
                threads=threads,
            )
            filename = f"{key}.aln.fasta"
            write_fasta(
                zip([identifier for identifier, _sequence in selected], result.aligned, strict=True),
                aligned_dir / filename,
            )

            columns = len(result.aligned[0]) if result.aligned else 0
            gap_fraction = (
                sum(row.count("-") for row in result.aligned) / (len(result.aligned) * columns) if columns else 0.0
            )
            report[key] = {
                "n_available": available,
                "n_aligned": len(selected),
                "length_range": [low, high],
                "n_columns": columns,
                # Reported so a reader can judge the alignment rather than assume it.
                # Above roughly 0.8 the matrix is mostly gaps and carries little signal.
                "gap_fraction": round(gap_fraction, 4),
                "backend": result.backend,
                "path": f"msa_aligned/{filename}",
            }
    return report
