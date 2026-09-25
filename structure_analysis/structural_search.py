"""Structural homology search over AlphaFold models, via Foldseek.

Sequence-based transfer finds donors that a profile HMM can still recognise. Beyond that
point - the twilight zone, roughly 20-25% identity - structure is conserved long after
sequence is not, and two proteins with the same fold and the same function can share almost
no recognisable sequence. Those are exactly the donors a J-domain protein from an unstudied
organism needs, and exactly the ones sequence search cannot supply.

Why Foldseek rather than Dali
-----------------------------
Dali is the reference method and would give slightly better alignments, but it is far too
slow for this: the public server is rate-limited and DaliLite scales to hundreds of
comparisons, not to a 142,948-model set searched against itself. Foldseek discretises local
backbone geometry into a structural alphabet and then runs the search as a sequence problem,
which makes it several orders of magnitude faster at comparable sensitivity. It was built
for AlphaFold-scale databases, which is what this is.

Foldseek is vendored under ``vendor/foldseek`` for the same reason MAFFT is: this cluster
exposes it only through licensed module trees that refuse to load inside a batch job.

What this adds over the sequence route
--------------------------------------
Structural and sequence search find overlapping but different donors, so the pipeline runs
both and records which route supplied each one. A term transferred from a structural donor
alone is a stronger claim than one from a sequence donor alone - it survived the loss of
sequence similarity - but it is also the one most in need of the provenance this module
attaches to it.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from domain_layout.progress import progress

logger = logging.getLogger(__name__)

FOLDSEEK_EXECUTABLE = "foldseek"

# Foldseek E-value ceiling for a structural match to be reported. Looser than the sequence
# threshold because structural alignment scores differently, and because the point of this
# route is to reach donors sequence cannot.
DEFAULT_EVALUE = 1e-5

# Minimum template modelling score. TM-score above 0.5 is the conventional threshold for
# two proteins sharing a fold; below it the alignment may be locally reasonable and
# globally meaningless.
MIN_TM_SCORE = 0.5

# Foldseek sensitivity. Higher finds more remote relationships at more cost; 9.5 is the
# setting Foldseek's own documentation recommends for remote homology rather than for
# speed, which is the regime that matters here.
SENSITIVITY = "9.5"

# Wall-clock ceiling for one search, so a pathological query cannot hold a job open.
DEFAULT_TIMEOUT_SECONDS = 7200

# Parts in an AlphaFold model filename, "AF-<accession>-F1-model_v4", before the accession
# can be trusted to be the second field.
_MODEL_NAME_PARTS = 2

# Sequence identity below which a pairwise sequence method is unreliable - the twilight
# zone. A confident structural match here is a donor the sequence route could not supply.
TWILIGHT_ZONE_IDENTITY = 0.25


class NoReadableStructuresError(RuntimeError):
    """Raised when a structure directory holds nothing Foldseek can parse."""


class InsufficientStagingSpaceError(RuntimeError):
    """Raised when decompressing the requested models would not fit."""


@dataclass(frozen=True, slots=True)
class StructuralHit:
    """One structural match between a query and a database model."""

    query: str
    target: str
    e_value: float
    tm_score: float
    alignment_length: int
    sequence_identity: float

    @property
    def is_remote(self) -> bool:
        """Whether this match lies beyond what sequence search would find.

        Below about 25% identity a pairwise sequence method is unreliable, so a confident
        structural match there is a donor the sequence route could not have supplied.
        """
        return self.sequence_identity < TWILIGHT_ZONE_IDENTITY

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "query": self.query,
            "target": self.target,
            "e_value": self.e_value,
            "tm_score": round(self.tm_score, 4),
            "alignment_length": self.alignment_length,
            "sequence_identity": round(self.sequence_identity, 4),
            "beyond_sequence_search": self.is_remote,
        }


def foldseek_available(executable: str = FOLDSEEK_EXECUTABLE) -> bool:
    """Whether the Foldseek binary can be found."""
    return shutil.which(executable) is not None or Path(executable).is_file()


def _accession_from_model(name: str) -> str:
    """Recover a UniProt accession from an AlphaFold model filename."""
    stem = Path(name).name
    for suffix in (".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        stem = stem.removesuffix(suffix)
    parts = stem.split("-")
    # Model filenames look like AF-<accession>-F1-model_v4.
    return parts[1] if len(parts) > _MODEL_NAME_PARTS and parts[0] == "AF" else stem


def parse_alignment(text: str) -> list[StructuralHit]:
    """Parse Foldseek tabular output into hits.

    Columns are requested explicitly when the search runs, so this depends on the format
    string in :func:`search_structures` rather than on Foldseek's default ordering.
    """
    hits: list[StructuralHit] = []
    for line in text.splitlines():
        fields = line.rstrip("\n").split("\t")
        expected_fields = 6
        if len(fields) < expected_fields:
            continue
        query, target, evalue, identity, length, tm_score = fields[:expected_fields]
        try:
            hits.append(
                StructuralHit(
                    query=_accession_from_model(query),
                    target=_accession_from_model(target),
                    e_value=float(evalue),
                    tm_score=float(tm_score),
                    alignment_length=int(float(length)),
                    sequence_identity=float(identity),
                ),
            )
        except ValueError:
            continue
    return hits


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Thresholds and resources for one structural search."""

    executable: str = FOLDSEEK_EXECUTABLE
    e_value: float = DEFAULT_EVALUE
    min_tm: float = MIN_TM_SCORE
    threads: int = 1
    timeout: int = DEFAULT_TIMEOUT_SECONDS


DEFAULT_SEARCH_OPTIONS = SearchOptions()


# Extensions Foldseek can read directly. Anything gzipped has to be staged first: its
# directory scan silently skips compressed files, and the run then reports "0 hits" - a
# statement about the search that is really a statement about the input.
READABLE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".ent")

# Decompressed size relative to gzipped, measured on AlphaFold PDB models.
_DECOMPRESSION_FACTOR = 4
# How many files to sample when estimating the mean compressed size.
_SIZE_SAMPLE = 200
# Refuse to fill more than this share of what is free, leaving room for everything else
# writing to the same filesystem.
_SAFETY_MARGIN = 0.8

# Hard ceiling on staged bytes, independent of what the filesystem claims is free.
#
# Necessary because the quota cannot be read here: `lfs` is not installed on this cluster,
# so `_lustre_quota_remaining` always returns None and the guard falls back to filesystem
# free space - which reported hundreds of terabytes while a group quota of a few gigabytes
# was already exhausted. That mismatch killed unrelated jobs three times. An explicit
# ceiling is worse engineering than reading the real limit and is the only thing that
# actually holds when the real limit is unreadable.
DEFAULT_MAX_STAGED_BYTES = 8 * 10**9

# Seconds to wait on the quota command before falling back to filesystem free space.
_QUOTA_TIMEOUT_SECONDS = 20


def available_space(path: Path) -> int:
    """Bytes actually writable at a path, respecting a group quota if one applies.

    ``shutil.disk_usage`` reports the *filesystem*, which on a shared cluster is the wrong
    number by orders of magnitude: it said 397 TB free while the group quota was already
    exhausted, so the staging guard passed and then filled the quota anyway - killing jobs
    that had nothing to do with it. Lustre's own quota is consulted first and the smaller
    of the two is returned.
    """
    filesystem = shutil.disk_usage(path).free
    quota = _lustre_quota_remaining(path)
    return min(filesystem, quota) if quota is not None else filesystem


def _lustre_quota_remaining(path: Path) -> int | None:
    """Bytes left under the group quota covering a path, or None if there is no quota."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["lfs", "quota", "-q", "-g", _group_of(path), str(path)],  # noqa: S607 - resolved via PATH by design
            capture_output=True,
            text=True,
            timeout=_QUOTA_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    # Columns: filesystem, used(kB), quota(kB), limit(kB), ...
    fields = completed.stdout.split()
    try:
        used_kb = int(fields[1].rstrip("*"))
        limit_kb = int(fields[2])
    except (IndexError, ValueError):
        return None
    if limit_kb <= 0:
        return None
    return max(0, (limit_kb - used_kb) * 1024)


def _group_of(path: Path) -> str:
    """Owning group name of a path, used to ask about the right quota."""
    try:
        import grp  # noqa: PLC0415 - unavailable on non-POSIX, imported where used

        return grp.getgrgid(path.stat().st_gid).gr_name
    except (OSError, KeyError, ImportError):
        return ""


def readable_structure_count(directory: Path) -> int:
    """How many files in a directory Foldseek could actually read."""
    if not directory.is_dir():
        return 0
    return sum(1 for item in directory.iterdir() if item.suffix.lower() in READABLE_SUFFIXES)


def stage_structures(  # noqa: PLR0913 - source, destination, format, and three independent bounds
    source: Path,
    destination: Path,
    *,
    suffix: str = ".pdb",
    limit: int | None = None,
    accessions: Sequence[str] | None = None,
    max_bytes: int = DEFAULT_MAX_STAGED_BYTES,
) -> int:
    """Decompress gzipped models into a directory Foldseek can read.

    AlphaFold models arrive as ``.pdb.gz`` / ``.cif.gz``. Foldseek's ``createdb`` skips them
    without error and builds an empty database, so a two-hour all-versus-all returned zero
    hits and looked like a biological result rather than an unread input.

    One format is staged, not both: the same model as ``.pdb`` and ``.cif`` would enter the
    database twice and every protein would score a perfect self-hit against its own twin.
    """
    destination.mkdir(parents=True, exist_ok=True)
    pattern = f"*{suffix}.gz"
    sources = sorted(source.glob(pattern))
    if accessions is not None:
        # Stage only the chosen representatives. Filtering here rather than after
        # decompression is the whole point: writing 142,948 models and then ignoring most
        # of them is what exhausted the quota.
        wanted = set(accessions)
        sources = [item for item in sources if _accession_from_model(item.name) in wanted]
    if limit is not None:
        sources = sources[:limit]

    # Check the space first. Decompressed AlphaFold models run roughly four times their
    # gzipped size, and staging the full set without checking exhausted a shared group
    # quota - which killed an unrelated GPU job mid-write and cost its whole run. Failing
    # here costs seconds; failing after 50,000 files costs everything running beside it.
    needed = int(sum(item.stat().st_size for item in sources[:_SIZE_SAMPLE]) / max(1, min(len(sources), _SIZE_SAMPLE)))
    needed *= len(sources) * _DECOMPRESSION_FACTOR
    free = available_space(destination)
    ceiling = min(int(free * _SAFETY_MARGIN), max_bytes)
    if needed > ceiling:
        message = (
            f"staging {len(sources)} model(s) needs roughly {needed / 1e9:.1f} GB, over the "
            f"{ceiling / 1e9:.1f} GB ceiling at {destination}. Pass a smaller --max-structures, "
            f"or raise max_bytes if the quota genuinely allows it. The ceiling is explicit "
            f"because the group quota cannot be read on every cluster, and filesystem free "
            f"space is not a substitute - the two differed by five orders of magnitude here."
        )
        raise InsufficientStagingSpaceError(message)

    staged = 0
    for item in progress(sources, description="staging structures", unit="model"):
        target = destination / item.name[: -len(".gz")]
        if target.exists():
            staged += 1
            continue
        with gzip.open(item, "rb") as handle, target.open("wb") as out:
            shutil.copyfileobj(handle, out)
        staged += 1
    logger.info("staged %s decompressed model(s) into %s", staged, destination)
    return staged


def search_structures(
    query_dir: Path,
    target_dir: Path,
    *,
    options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
) -> list[StructuralHit]:
    """Search every model in ``query_dir`` against every model in ``target_dir``.

    Returns hits above the E-value and TM-score thresholds, or an empty list when Foldseek
    is unavailable or the search fails - the pipeline degrades to sequence-only transfer
    rather than aborting.
    """
    executable, e_value = options.executable, options.e_value
    min_tm, threads, timeout = options.min_tm, options.threads, options.timeout

    if not foldseek_available(executable):
        logger.warning("foldseek not found; structural donors will not be searched.")
        return []
    if not query_dir.is_dir() or not target_dir.is_dir():
        return []

    # Distinguish "the search found nothing" from "the search had nothing to read". The
    # two are indistinguishable downstream, and reporting the second as the first turned an
    # unread directory of gzipped models into an apparent biological result.
    readable = readable_structure_count(query_dir)
    if readable == 0:
        compressed = sum(1 for item in query_dir.iterdir() if item.name.endswith(".gz"))
        message = (
            f"no structures Foldseek can read in {query_dir} "
            f"({compressed} gzipped file(s) present). Stage them with stage_structures() first."
        )
        raise NoReadableStructuresError(message)

    with tempfile.TemporaryDirectory(prefix="foldseek_") as tmp:
        output = Path(tmp) / "aln.tsv"
        command = [
            executable,
            "easy-search",
            str(query_dir),
            str(target_dir),
            str(output),
            str(Path(tmp) / "tmp"),
            "-e",
            str(e_value),
            "-s",
            SENSITIVITY,
            "--threads",
            str(max(1, threads)),
            "--format-output",
            "query,target,evalue,fident,alnlen,alntmscore",
        ]
        try:
            subprocess.run(  # noqa: S603 - fixed argv, no shell, paths are ours
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("foldseek timed out after %ss; falling back to sequence donors only.", timeout)
            return []
        except (subprocess.CalledProcessError, OSError):
            logger.warning("foldseek failed; falling back to sequence donors only.", exc_info=True)
            return []

        if not output.is_file():
            return []
        hits = parse_alignment(output.read_text(encoding="utf-8"))

    # Self-matches carry no information, and a fold match below the conventional threshold
    # may be locally plausible and globally meaningless.
    return [hit for hit in hits if hit.query != hit.target and hit.tm_score >= min_tm]


def best_donors(
    hits: Sequence[StructuralHit],
    donors: Mapping[str, object],
    *,
    max_per_query: int = 10,
) -> dict[str, list[StructuralHit]]:
    """Group hits by query, keeping only matches to proteins eligible to donate."""
    grouped: dict[str, list[StructuralHit]] = {}
    for hit in hits:
        if hit.target not in donors:
            continue
        grouped.setdefault(hit.query, []).append(hit)
    for query, items in grouped.items():
        items.sort(key=lambda item: -item.tm_score)
        grouped[query] = items[:max_per_query]
    return grouped
