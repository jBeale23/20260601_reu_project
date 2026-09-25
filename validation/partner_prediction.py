"""Predicting which Hsp70 a protein partners, and checking whether the prediction is right.

Everything up to here measured *association*: whether a grouping's labels and a readout's
labels are statistically related. That is the right question for "does this carving track
biology", and the wrong one for "can you use it". A scheme can carry information about
partner identity and still be useless as a predictor, because information is symmetric and
prediction is not.

This asks the usable question. Assign every protein the commonest Hsp70 partner among the
*other* members of its group, then check that assignment against the partner it actually has.
The answer is an accuracy, in the same units as a coin flip, and it can be wrong.

Two controls decide whether the number means anything.

**Predictions are out of fold.** A protein is never predicted by a rule fitted on itself. The
naive version - take each group's modal partner and score it on the same proteins - is
memorisation, and on a twelve-group partition it would report near-perfect accuracy while
predicting nothing.

**Folds are blocked.** J-domain proteins come in orthologue families, and two proteins from
the same genus are often 95% identical. Random folds let a protein be predicted by its own
near-twin, which measures how redundant the dataset is rather than how well the scheme
generalises. Blocking is what makes the number an estimate of performance on an organism
nobody has studied - which is the entire point of the project.

The blocking unit is a choice, and the useful ladder is species, then genus, then
sequence-identity clusters. Species and genus are not two separate controls to combine:
species nest inside genera, so blocking genus already blocks species and blocking "both" is
just blocking genus. Running the ladder and watching accuracy fall is how much leakage each
level was hiding.

The baseline that matters is the majority partner. If 60% of proteins partner DnaK, a scheme
that scores 60% has learned nothing, and a scheme scoring 62% has learned almost nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Fewest proteins in a fold's training half before a group's modal partner is worth trusting.
MIN_GROUP_TRAIN = 5

# Folds for the blocked split. Five keeps each training half large while leaving a fifth
# or so of blocks out each time.
DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class PredictionResult:
    """How well one scheme predicts partner identity, out of fold."""

    scheme: str
    n_predicted: int
    n_correct: int
    n_abstained: int
    majority_accuracy: float
    n_groups: int

    @property
    def accuracy(self) -> float:
        """Share of predictions that were right."""
        return self.n_correct / self.n_predicted if self.n_predicted else 0.0

    @property
    def lift_over_majority(self) -> float:
        """Accuracy above always guessing the commonest partner.

        The meaningful quantity: a scheme that cannot beat the majority guess has learned
        nothing about partner identity, however good its raw accuracy looks.
        """
        return self.accuracy - self.majority_accuracy

    @property
    def coverage(self) -> float:
        """Share of proteins the scheme was willing to predict at all."""
        total = self.n_predicted + self.n_abstained
        return self.n_predicted / total if total else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Render for the report."""
        return {
            "scheme": self.scheme,
            "n_groups": self.n_groups,
            "n_predicted": self.n_predicted,
            "n_abstained": self.n_abstained,
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "majority_accuracy": round(self.majority_accuracy, 4),
            "lift_over_majority": round(self.lift_over_majority, 4),
        }


def blocked_folds(
    accessions: Sequence[str],
    block_key: Mapping[str, str],
    *,
    n_folds: int = DEFAULT_FOLDS,
) -> list[set[str]]:
    """Split accessions into folds that never share a blocking key.

    The key is whatever unit should not be split across the train/test boundary - species,
    genus, or a sequence-identity cluster. Genus subsumes species, since species nest inside
    genera, so blocking both is the same as blocking genus; the meaningful ladder runs
    species, then genus, then identity clustering, each stricter than the last.

    Blocks are assigned largest-first, each going to the fold currently holding fewest
    proteins. That keeps folds comparable in size despite block sizes ranging from one
    protein to a hundred and seventy.
    """
    by_block: dict[str, list[str]] = {}
    for accession in accessions:
        by_block.setdefault(block_key.get(accession, "unknown"), []).append(accession)

    folds: list[set[str]] = [set() for _ in range(max(1, n_folds))]
    for _block, members in sorted(by_block.items(), key=lambda kv: -len(kv[1])):
        smallest = min(folds, key=len)
        smallest.update(members)
    return folds


def predict_out_of_fold(
    partition: Mapping[str, str],
    partner: Mapping[str, str],
    block_key: Mapping[str, str],
    *,
    scheme: str,
    n_folds: int = DEFAULT_FOLDS,
) -> PredictionResult:
    """Predict each protein's partner from its group, using only other folds to fit.

    A group whose training half holds fewer than ``MIN_GROUP_TRAIN`` proteins abstains rather
    than guessing from one or two examples. Abstentions are reported rather than scored as
    errors, because a scheme that declines to predict is behaving differently from one that
    predicts wrongly, and collapsing them would hide which is happening.
    """
    shared = sorted(set(partition) & set(partner))
    if not shared:
        return PredictionResult(scheme, 0, 0, 0, 0.0, 0)

    folds = blocked_folds(shared, block_key, n_folds=n_folds)
    correct = predicted = abstained = 0

    for held_out in folds:
        train = [a for a in shared if a not in held_out]
        modal: dict[str, str] = {}
        for group, members in _by_group(train, partition).items():
            counts = Counter(partner[a] for a in members)
            if sum(counts.values()) >= MIN_GROUP_TRAIN:
                modal[group] = counts.most_common(1)[0][0]

        for accession in held_out:
            guess = modal.get(partition[accession])
            if guess is None:
                abstained += 1
                continue
            predicted += 1
            correct += guess == partner[accession]

    # The majority baseline is computed the same way - out of fold - so the comparison is
    # like for like rather than against a baseline fitted on everything.
    majority_correct = majority_total = 0
    for held_out in folds:
        train = [a for a in shared if a not in held_out]
        if not train:
            continue
        guess = Counter(partner[a] for a in train).most_common(1)[0][0]
        for accession in held_out:
            majority_total += 1
            majority_correct += guess == partner[accession]

    return PredictionResult(
        scheme=scheme,
        n_predicted=predicted,
        n_correct=correct,
        n_abstained=abstained,
        majority_accuracy=majority_correct / majority_total if majority_total else 0.0,
        n_groups=len({partition[a] for a in shared}),
    )


def _by_group(accessions: Sequence[str], partition: Mapping[str, str]) -> dict[str, list[str]]:
    """Members of each group, for the accessions given."""
    out: dict[str, list[str]] = {}
    for accession in accessions:
        out.setdefault(partition[accession], []).append(accession)
    return out


def group_partner_profile(
    partition: Mapping[str, str],
    partner: Mapping[str, str],
    *,
    min_members: int = MIN_GROUP_TRAIN,
) -> dict[str, dict[str, object]]:
    """Which partner dominates each group, and how strongly.

    This is where a mechanism would become visible. An accuracy says a scheme predicts; it
    does not say what it predicts, and a carving is only interpretable once you can read off
    which group goes with which chaperone. A group that is 90% one partner is a hypothesis
    about that group's biology; one split evenly across four is not.
    """
    profile: dict[str, dict[str, object]] = {}
    for group, members in _by_group(sorted(set(partition) & set(partner)), partition).items():
        counts = Counter(partner[a] for a in members)
        total = sum(counts.values())
        if total < min_members:
            continue
        top, top_count = counts.most_common(1)[0]
        profile[group] = {
            "n": total,
            "dominant_partner": top,
            "purity": round(top_count / total, 4),
            "top_partners": dict(counts.most_common(4)),
        }
    return profile
