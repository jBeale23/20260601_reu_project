"""Profile-HMM homology layer, built on HMMER3 via :mod:`pyhmmer`.

This closes the loop on the MSA half of the routing: MAFFT aligns a domain family
(:mod:`domain_layout.msa`), HMMER turns that alignment into a profile hidden Markov model,
and any protein can then be scored against it. A profile HMM is strictly more sensitive
than the pairwise similarity used elsewhere here, because it learns a per-position
substitution and gap model from the family instead of applying one fixed matrix
everywhere.

Why this exists as a *baseline* rather than as the classifier
-------------------------------------------------------------
Profile HMMs are the standard against which any new homology-detection method is
measured, and they are what InterPro's own signatures are built from - so a rule-based or
learned classifier that cannot beat ``hmmsearch`` on the same data has not earned its
complexity. :mod:`validation` therefore scores the HMM baseline next to the layout
classifier on identical proteins.

Their known ceiling is equally important to state: profile HMMs are built on positional
conservation, so they lose power exactly where this project is most interested - highly
diverged J-domain proteins in non-model organisms, and the unalignable regions that carry
much of the class signal. That is a reason to measure against them, not a reason to skip
them.

Implementation note
-------------------
``pyhmmer`` is HMMER3 itself compiled as a Python extension, not a reimplementation, so
results match the ``hmmbuild`` / ``hmmsearch`` binaries. It is used in preference to
shelling out because it needs no binary on ``PATH``, installs from PyPI on any platform
this project targets, and returns structured hits instead of a tabular format to parse.
The layer degrades to unavailable rather than failing when it is not installed, matching
every other optional backend here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

BACKEND_PYHMMER = "pyhmmer"
BACKEND_NONE = "unavailable"

# Reported alongside every hit. HMMER's bit score is comparable across queries of the same
# profile; the E-value is not comparable across searches of different database sizes, so
# both are kept and the caller is told which to use.
DEFAULT_E_VALUE_CUTOFF = 1e-3

# A profile built from one or two sequences encodes almost no positional variation and
# behaves like a glorified pairwise alignment, so families below this are not modelled.
MIN_SEQUENCES_FOR_PROFILE = 3


def hmmer_available() -> bool:
    """Whether ``pyhmmer`` can be imported."""
    return find_spec("pyhmmer") is not None


def active_backend() -> str:
    """Report which homology backend a run would use."""
    return BACKEND_PYHMMER if hmmer_available() else BACKEND_NONE


@dataclass(frozen=True, slots=True)
class ProfileHit:
    """One protein's best match against one profile HMM."""

    target: str
    profile: str
    score: float
    e_value: float

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "target": self.target,
            "profile": self.profile,
            "score": round(self.score, 3),
            "e_value": self.e_value,
        }


@dataclass(frozen=True, slots=True)
class FamilyProfile:
    """A profile HMM built from one aligned domain family."""

    name: str
    n_sequences: int
    n_positions: int
    # The opaque pyhmmer HMM object. Untyped because pyhmmer is an optional import.
    model: object

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the profile's shape (not the model itself)."""
        return {
            "name": self.name,
            "n_sequences": self.n_sequences,
            "n_positions": self.n_positions,
        }


def build_profile(name: str, aligned_sequences: Sequence[str]) -> FamilyProfile | None:
    """Build a profile HMM from an existing alignment.

    Args:
        name: Family name recorded on the profile.
        aligned_sequences: Equal-length gapped rows, as produced by
            :func:`domain_layout.msa.align_sequences`.

    Returns:
        The profile, or ``None`` when pyhmmer is unavailable, the family is too small, or
        the rows are not a valid alignment.
    """
    if not hmmer_available():
        return None
    if len(aligned_sequences) < MIN_SEQUENCES_FOR_PROFILE:
        return None
    if len({len(row) for row in aligned_sequences}) != 1:
        logger.warning("%s: rows are not equal length; not an alignment, skipping profile.", name)
        return None

    from pyhmmer.easel import Alphabet, TextMSA, TextSequence  # noqa: PLC0415
    from pyhmmer.plan7 import Background, Builder  # noqa: PLC0415

    alphabet = Alphabet.amino()
    rows = [
        TextSequence(name=f"{name}_{index}".encode(), sequence=row.upper())
        for index, row in enumerate(aligned_sequences)
    ]
    try:
        msa = TextMSA(name=name.encode(), sequences=rows).digitize(alphabet)
        model, _profile, _optimized = Builder(alphabet).build_msa(msa, Background(alphabet))
    except (ValueError, RuntimeError):
        logger.warning("%s: hmmbuild failed on this alignment.", name, exc_info=True)
        return None

    return FamilyProfile(
        name=name,
        n_sequences=len(aligned_sequences),
        n_positions=int(model.M),
        model=model,
    )


def _decode(value: object) -> str:
    """Pyhmmer returns names as bytes on some versions and str on others."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _best_by_target(hits: object, profile_name: str) -> dict[str, ProfileHit]:
    """Collapse one profile's hits to the best-scoring hit per target."""
    best: dict[str, ProfileHit] = {}
    for hit in hits:  # type: ignore[union-attr]
        target = _decode(hit.name)
        candidate = ProfileHit(
            target=target,
            profile=profile_name,
            score=float(hit.score),
            e_value=float(hit.evalue),
        )
        existing = best.get(target)
        if existing is None or candidate.score > existing.score:
            best[target] = candidate
    return best


def search_profiles(
    profiles: Sequence[FamilyProfile],
    sequences: Mapping[str, str],
    *,
    e_value_cutoff: float = DEFAULT_E_VALUE_CUTOFF,
    cpus: int = 0,
) -> list[dict[str, ProfileHit]]:
    """Score sequences against many profiles at once, one result per profile in order.

    Searching profiles one at a time re-encodes the entire target set for every profile.
    That is invisible on a handful of families and ruinous at scale: the functional
    transfer run searches 198 donor profiles against 181,328 targets, so the per-profile
    form spends 36 million sequence digitisations to do 198 searches' worth of work. Here
    the targets are digitised once and reused, and pyhmmer threads the searches across
    cores, which is what turns that job from a multi-day grind into a short one.

    Args:
        profiles: Profiles to search with.
        sequences: Targets, keyed by accession.
        e_value_cutoff: Reporting threshold.
        cpus: Worker threads; 0 lets pyhmmer use every available core.

    Returns:
        One mapping of accession to best hit per profile, aligned to ``profiles``.
        Results are positional rather than keyed by profile name because names are not
        guaranteed unique and a name collision would silently drop a profile's hits.
    """
    if not hmmer_available() or not sequences or not profiles:
        return [{} for _ in profiles]

    import pyhmmer  # noqa: PLC0415
    from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequence  # noqa: PLC0415

    alphabet = Alphabet.amino()
    targets: list[object] = []
    for accession, sequence in sequences.items():
        if not sequence:
            continue
        targets.append(TextSequence(name=accession.encode(), sequence=sequence.upper()).digitize(alphabet))
    if not targets:
        return [{} for _ in profiles]

    block = DigitalSequenceBlock(alphabet, targets)
    try:
        all_hits = list(
            pyhmmer.hmmsearch(
                [profile.model for profile in profiles],
                block,
                E=e_value_cutoff,
                cpus=cpus,
            )
        )
    except (ValueError, RuntimeError):
        logger.warning("batched hmmsearch failed over %d profiles.", len(profiles), exc_info=True)
        return [{} for _ in profiles]

    # pyhmmer yields one TopHits per query in query order; anything else means the
    # correspondence to profiles is not what this function promises, so do not guess.
    if len(all_hits) != len(profiles):
        logger.warning(
            "hmmsearch returned %d result sets for %d profiles; discarding to avoid mislabelling hits.",
            len(all_hits),
            len(profiles),
        )
        return [{} for _ in profiles]

    return [_best_by_target(hits, profile.name) for profile, hits in zip(profiles, all_hits, strict=True)]


def search_profile(
    profile: FamilyProfile,
    sequences: Mapping[str, str],
    *,
    e_value_cutoff: float = DEFAULT_E_VALUE_CUTOFF,
) -> dict[str, ProfileHit]:
    """Score sequences against one profile, keeping each target's best hit.

    Returns:
        Mapping of accession to its hit. Sequences that do not clear ``e_value_cutoff``
        are absent rather than present with a null score, so a caller cannot mistake
        "did not match" for "matched at zero".
    """
    return search_profiles([profile], sequences, e_value_cutoff=e_value_cutoff)[0]


def classify_by_best_profile(
    profiles: Sequence[FamilyProfile],
    sequences: Mapping[str, str],
    profile_classes: Mapping[str, str],
    *,
    e_value_cutoff: float = DEFAULT_E_VALUE_CUTOFF,
) -> dict[str, str]:
    """Assign each sequence the class of its highest-scoring profile.

    This is the homology baseline the layout classifier is measured against: pure
    profile-HMM nearest-family assignment, with no architectural reasoning at all.

    Sequences matching no profile are left unassigned rather than defaulted to a class.
    Defaulting would quietly convert the baseline's failures into successes whenever the
    default happened to be the majority class, which is exactly the comparison the
    validation suite exists to make honest.
    """
    scores: dict[str, ProfileHit] = {}
    for per_profile in search_profiles(profiles, sequences, e_value_cutoff=e_value_cutoff):
        for target, hit in per_profile.items():
            existing = scores.get(target)
            if existing is None or hit.score > existing.score:
                scores[target] = hit

    return {target: profile_classes[hit.profile] for target, hit in scores.items() if hit.profile in profile_classes}
