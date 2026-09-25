"""Profile-HMM baseline for the classification task, cross-validated.

A rule-based or learned classifier only earns its complexity by beating the standard
method. For remote protein classification that method is the profile HMM, and it is what
InterPro's own signatures are built from - so if ``hmmsearch`` against per-class profiles
matches the layout classifier, the layout reasoning has added nothing.

Avoiding the leak that would make this baseline meaningless
-----------------------------------------------------------
The obvious implementation builds one profile per class from the labelled proteins and
then classifies those same proteins. That baseline is not weak, it is *invalid*: each
protein contributed to the profile it is later scored against, so its own residues are
part of the model that recognises it. Reported accuracy would approach one and the
comparison would be meaningless.

This module therefore uses stratified k-fold cross-validation. Profiles are built from the
training folds only, every protein is predicted exactly once while held out, and the
predictions returned are out-of-fold throughout. Stratification keeps the rarest class
present in every training split, which matters here because class A is a sixth of the
labelled set.

A second leak is measured rather than merely noted. Close homologs of a held-out protein
sit in its training folds when the split is random, and that does **not** affect every
model equally: a profile HMM is built from sequence and can recognise a near-identical
relative, while a rule system reading domain architecture gains far less from one. A
random-split comparison is therefore biased *towards* the homology baseline.

So both splits are reported. :func:`profile_hmm_predictions` cross-validates at random;
:func:`genus_blocked_predictions` keeps every protein from a genus in the same fold, so no
held-out protein has a same-genus relative in training. The gap between the two is the
measure of how much of a score is memorised homology, and it is the genus-blocked number
that speaks to a novel organism.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_layout import hmmer as hmmer_backend
from domain_layout import msa as msa_backend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

BASELINE_NAME = "profile_hmm_cv"

# Folds for the cross-validation. Five keeps each training split large enough to model the
# rarest class while still holding out a fifth of the data at a time.
DEFAULT_N_FOLDS = 5

# Sequences aligned per class per fold. A profile gains little past a few hundred members
# and MAFFT's cost grows faster than linearly, so larger classes are sub-sampled.
MAX_SEQUENCES_PER_PROFILE = 300


def genus_of(organism_name: str) -> str:
    """First token of a binomial name, used as the blocking unit."""
    token = (organism_name or "").strip().split(" ")[0] if organism_name else ""
    return token or "unknown"


def _genus_blocked_folds(
    labels: Mapping[str, str],
    organisms: Mapping[str, str],
    *,
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    """Split accessions so every protein of a genus lands in the same fold.

    Blocking on genus is a coarse stand-in for clustering by sequence identity, which
    would need an external tool. It is conservative in the useful direction: same-genus
    proteins are usually close homologs, so removing them from training removes the
    easiest cases rather than a random sample of them.

    Genera are assigned to whichever fold is currently smallest, which keeps folds
    comparable in size even though genus sizes are highly uneven.
    """
    by_genus: dict[str, list[str]] = defaultdict(list)
    for accession in sorted(labels):
        by_genus[genus_of(organisms.get(accession, ""))].append(accession)

    # Deterministic, seeded ordering for reproducibility; not a security context.
    rng = random.Random(seed)  # noqa: S311
    genera = sorted(by_genus, key=lambda name: (-len(by_genus[name]), name))
    rng.shuffle(genera)
    genera.sort(key=lambda name: -len(by_genus[name]))

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for genus in genera:
        smallest = min(range(n_folds), key=lambda index: len(folds[index]))
        folds[smallest].extend(by_genus[genus])
    return folds


def _stratified_folds(
    labels: Mapping[str, str],
    *,
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    """Split accessions into folds that each preserve the class distribution."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for accession, label in sorted(labels.items()):
        by_class[label].append(accession)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Deterministic, seeded assignment for reproducibility; not a security context.
    rng = random.Random(seed)  # noqa: S311
    for label in sorted(by_class):
        members = list(by_class[label])
        rng.shuffle(members)
        # Deal round-robin so every fold receives a near-equal share of each class.
        for position, accession in enumerate(members):
            folds[position % n_folds].append(accession)
    return folds


def _class_profiles(
    training: Sequence[str],
    labels: Mapping[str, str],
    sequences: Mapping[str, str],
    *,
    msa_backend_name: str,
    seed: int,
) -> tuple[list[object], dict[str, str]]:
    """Build one profile HMM per class from the training accessions."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for accession in training:
        sequence = sequences.get(accession, "")
        if sequence:
            by_class[labels[accession]].append(sequence)

    profiles: list[object] = []
    profile_classes: dict[str, str] = {}
    for label in sorted(by_class):
        members = by_class[label]
        if len(members) > MAX_SEQUENCES_PER_PROFILE:
            rng = random.Random(seed)  # noqa: S311 - reproducible sub-sampling
            members = [members[index] for index in sorted(rng.sample(range(len(members)), MAX_SEQUENCES_PER_PROFILE))]

        alignment = msa_backend.align_sequences(members, backend=msa_backend_name)
        profile_name = f"class_{label}"
        profile = hmmer_backend.build_profile(profile_name, alignment.aligned)
        if profile is None:
            logger.info("class %s: too few training sequences for a profile in this fold", label)
            continue
        profiles.append(profile)
        profile_classes[profile_name] = label
    return profiles, profile_classes


def profile_hmm_predictions(
    labels: Mapping[str, str],
    sequences: Mapping[str, str],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    msa_backend_name: str = msa_backend.BACKEND_AUTO,
) -> dict[str, str]:
    """Out-of-fold profile-HMM class predictions for every labelled protein.

    Returns:
        Mapping of accession to predicted class. Proteins matching no profile while held
        out are absent rather than defaulted, so the baseline is neither credited nor
        penalised for a guess it did not make.
    """
    if not hmmer_backend.hmmer_available():
        logger.warning("pyhmmer is not installed; the profile-HMM baseline will not run.")
        return {}

    usable = {accession: label for accession, label in labels.items() if sequences.get(accession)}
    if len(usable) < n_folds:
        return {}

    folds = _stratified_folds(usable, n_folds=n_folds, seed=seed)
    predictions: dict[str, str] = {}
    for index, held_out in enumerate(folds):
        if not held_out:
            continue
        training = [accession for position, fold in enumerate(folds) if position != index for accession in fold]
        profiles, profile_classes = _class_profiles(
            training,
            usable,
            sequences,
            msa_backend_name=msa_backend_name,
            seed=seed,
        )
        if not profiles:
            continue
        assigned = hmmer_backend.classify_by_best_profile(
            profiles,
            {accession: sequences[accession] for accession in held_out},
            profile_classes,
        )
        predictions.update(assigned)
    return predictions


@dataclass(frozen=True, slots=True)
class LabelledProteins:
    """The labelled evaluation set: class, sequence, and organism per accession."""

    labels: Mapping[str, str]
    sequences: Mapping[str, str]
    organisms: Mapping[str, str]


def genus_blocked_predictions(
    proteins: LabelledProteins,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 0,
    msa_backend_name: str = msa_backend.BACKEND_AUTO,
) -> dict[str, str]:
    """Profile-HMM predictions with every genus confined to a single fold.

    The number that speaks to a novel organism. A drop from the random-split score is not
    a defect - it is the size of the homology advantage the random split was handing the
    baseline for free.
    """
    labels, sequences, organisms = proteins.labels, proteins.sequences, proteins.organisms
    if not hmmer_backend.hmmer_available():
        return {}

    usable = {accession: label for accession, label in labels.items() if sequences.get(accession)}
    if len(usable) < n_folds:
        return {}

    folds = _genus_blocked_folds(usable, organisms, n_folds=n_folds, seed=seed)
    predictions: dict[str, str] = {}
    for index, held_out in enumerate(folds):
        if not held_out:
            continue
        training = [accession for position, fold in enumerate(folds) if position != index for accession in fold]
        profiles, profile_classes = _class_profiles(
            training,
            usable,
            sequences,
            msa_backend_name=msa_backend_name,
            seed=seed,
        )
        if not profiles:
            continue
        predictions.update(
            hmmer_backend.classify_by_best_profile(
                profiles,
                {accession: sequences[accession] for accession in held_out},
                profile_classes,
            ),
        )
    return predictions


def baseline_notes(n_labelled: int, n_predicted: int) -> dict[str, object]:
    """Caveats that belong next to the baseline's score, not in a footnote."""
    return {
        "name": BASELINE_NAME,
        "n_labelled": n_labelled,
        "n_predicted": n_predicted,
        "n_unassigned": n_labelled - n_predicted,
        "backend": hmmer_backend.active_backend(),
        "cross_validation": f"stratified {DEFAULT_N_FOLDS}-fold, out-of-fold predictions only",
        "known_limitation": (
            "A random split leaves close homologs of a held-out protein in its training "
            "folds, and that does not affect every model equally: a profile HMM is built "
            "from sequence and recognises a near-identical relative, while a rule system "
            "reading domain architecture gains much less. The random-split comparison is "
            "therefore biased towards this baseline. The genus-blocked score is the one "
            "that speaks to a novel organism."
        ),
    }
