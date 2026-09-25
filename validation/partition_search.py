"""Searching for a grouping that genuinely beats A/B/C.

The sufficiency test can compare any two classifications against an independent readout.
That makes a search possible - generate candidate groupings from domain content, score each
one - and a search is exactly where a result can be manufactured. Try enough partitions and
one will beat A/B/C on any single readout by chance alone.

Three guards, and the third is the one that matters
---------------------------------------------------
**Correction across every candidate tested.** Not across the ones that looked promising:
the whole search is one family of tests and is corrected as one.

**A shape-matched null for each candidate.** A random partition with the same number of
groups of the same sizes. This asks whether a grouping's *content* explains a readout, or
merely its granularity - a five-group split beats a three-group split at almost anything.

**Discovery and validation readouts are disjoint.** Candidates are ranked on one readout and
the survivor is then required to beat A/B/C on a readout it was never scored against. A
partition selected for explaining Hsp70 partner choice has no claim on the interactome's
composition unless it earns it separately. Without this the first two guards still permit
selecting the candidate that best fits the noise in the readout used to select it.

A candidate that clears all three is worth looking at biologically. One that clears only the
first two is a hypothesis about the readout it was found on.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from validation.class_sufficiency import cramers_v
from validation.modality_information import mutual_information
from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PERMUTATIONS = 500
DEFAULT_SEED = 0
MIN_PROTEINS = 30
SIGNIFICANCE_LEVEL = 0.05

# A partition putting nearly everything in one group explains nothing, however good its
# score looks; a partition with a group per protein explains everything and means nothing.
MIN_GROUPS = 2
MAX_GROUPS = 12
MIN_GROUP_SHARE = 0.01

# Largest number of base features crossed into one candidate. Three is where the search
# stops paying: a fourth feature multiplies the cell count again, and almost nothing
# survives the minimum-share floor, so the extra candidates are all rejected before scoring.
MAX_COMBINATION_ORDER = 3

# Which statistic decides that a candidate beats the reference.
#
# ``cramers_v`` asks how close an association is to the maximum its table shape allows, and
# therefore penalises a finer partition even when the extra groups carry real information.
# Combinations of features are always finer than their parents, so a search for a winning
# *combination* scored this way can reject one that genuinely carries more signal.
#
# ``information`` asks only how much the partition tells you about the readout. Mutual
# information is monotone under refinement, so a combination cannot lose for fineness alone
# - but it also cannot lose for adding useless groups, which is why each estimate is taken
# above its own permutation null before being compared.
#
# The criterion must be fixed before looking at the results. Choosing it afterwards, once
# it is known which one favours the hypothesis, is the exact failure the discovery and
# validation split exists to prevent - so it is recorded in the report.
CRITERION_V = "cramers_v"
CRITERION_INFORMATION = "information"

# Smallest margin over the reference that counts as beating it, as a fraction of the
# reference's own score.
#
# Without a floor, "beats the reference" is any margin at all, and the permutation gate does
# not save it: at n = 9,300 every candidate clears its p-value, so significance stops
# discriminating. On the STRING readout that let a candidate confirm on a Cramer's V margin
# of 0.0004 - 0.2% of the reference - and seven of ten confirmations sat under 10%. A margin
# that small is a different sample, not a different classification.
MIN_RELATIVE_MARGIN = 0.10


def _usable(partition: Mapping[str, str]) -> bool:
    """Whether a partition is coarse enough to mean something and fine enough to test."""
    counts = Counter(partition.values())
    if not MIN_GROUPS <= len(counts) <= MAX_GROUPS:
        return False
    total = sum(counts.values())
    # Every group must hold a real share; a partition that is one big group plus dust is
    # a relabelling of "everything", and its score reflects the big group alone.
    return all(count >= total * MIN_GROUP_SHARE for count in counts.values())


def _systematic_combinations(
    built: Mapping[str, Mapping[str, str]],
    *,
    max_order: int = MAX_COMBINATION_ORDER,
) -> dict[str, dict[str, str]]:
    """Every combination of base features up to ``max_order``, not a hand-picked few.

    Interactions are where a combination beats its parts: a G/F-rich region mattering only
    in the absence of a C-terminal domain is invisible to either feature alone. The earlier
    version tested nine hand-chosen pairs, which cannot answer whether *some* combination
    beats the reference - it can only answer whether one of those nine does.

    Combinations are generated by crossing the base partitions and filtered by ``_usable``,
    which is what keeps the space from exploding: crossing three features with three groups
    each gives 27 cells, and almost all such partitions fail either the group-count ceiling
    or the minimum-share floor. The survivors are the ones with enough proteins in every
    cell to be worth a contingency table.

    The cost of searching wider is multiplicity, and that is paid for elsewhere: candidates
    are corrected together, and a candidate only counts as beating the reference if it does
    so on a readout it was never ranked against.
    """
    combined: dict[str, dict[str, str]] = {}
    names = sorted(built)
    for order in range(2, max_order + 1):
        for group in combinations(names, order):
            shared: set[str] = set(built[group[0]])
            for name in group[1:]:
                shared &= set(built[name])
            if len(shared) < MIN_PROTEINS:
                continue
            crossed = {accession: "_".join(built[name][accession] for name in group) for accession in shared}
            if _usable(crossed):
                combined["+".join(group)] = crossed
    logger.info("built %d usable combinations up to order %d", len(combined), max_order)
    return combined


def numeric_tertiles(
    features: Mapping[str, Mapping[str, str]],
    column: str,
    label: str,
) -> dict[str, dict[str, str]]:
    """Split one numeric column at tertiles fitted to the data, not at fixed cut points.

    Fitted cut points because these distributions are heavily skewed and nothing about a
    round number like 0.3 disorder is meaningful; a fixed threshold would put almost every
    protein on one side and destroy the very variation being tested.
    """
    values = {accession: float(row[column]) for accession, row in features.items() if _is_number(row.get(column))}
    if len(values) < MIN_PROTEINS:
        return {}
    ordered = sorted(values.values())
    low = ordered[len(ordered) // 3]
    high = ordered[2 * len(ordered) // 3]
    binned = {
        accession: (f"low_{label}" if value <= low else f"high_{label}" if value > high else f"mid_{label}")
        for accession, value in values.items()
    }
    return {f"{label}_tertile": binned} if _usable(binned) else {}


def _disorder_tertiles(features: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Disorder split at tertiles, kept under its original name for the default search."""
    return numeric_tertiles(features, "idr_fraction", "idr")


def _architecture_groups(features: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """The common domain layouts, with everything rarer pooled."""
    layouts = Counter(
        str(values.get("domain_family_layout", "")).strip()
        for values in features.values()
        if str(values.get("domain_family_layout", "")).strip()
    )
    common = {name for name, _count in layouts.most_common(8)}
    architecture = {
        accession: (layout if (layout := str(values.get("domain_family_layout", "")).strip()) in common else "other")
        for accession, values in features.items()
        if str(values.get("domain_family_layout", "")).strip()
    }
    return {"top_architectures": architecture} if _usable(architecture) else {}


# Numeric columns worth splitting, beyond the disorder fraction the original search used.
# Each is (column, short label). Deliberately excluded: anything named layout_predicted_*,
# layout_class_confidence, layout_novelty_score, novel_class_candidate, and
# shark_best_reference_class. Those are outputs of the classifier being tested, so a
# partition built from them would be scored against its own prediction.
WIDE_NUMERIC_COLUMNS = (
    ("idr_fraction", "idr"),
    ("mean_disorder", "disorder"),
    ("protein_length", "length"),
    ("n_entries", "entries"),
    ("n_msa_regions", "msa"),
    ("n_shark_regions", "shark"),
    ("shark_best_similarity", "similarity"),
)

# Binary architecture flags. These are the domain-content features a classifier could read
# off any sequence, in any organism, with no structure and no annotation.
BINARY_COLUMNS = (
    "has_dnaj_c",
    "has_zinc_finger_like",
    "has_gf_rich_region",
    "has_transmembrane",
    "has_signal_peptide",
    "has_hpd",
)


def candidate_partitions(
    features: Mapping[str, Mapping[str, str]],
    *,
    numeric_columns: Sequence[tuple[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Build candidate groupings from domain content.

    Every candidate is derived from architecture alone - domain presence, J-domain position,
    domain count, disorder - because a grouping that needs functional data to define it
    cannot then be tested against functional data.

    Args:
        features: One row of feature columns per accession.
        numeric_columns: Which numeric columns to split at tertiles, as (column, label).
            Defaults to disorder alone, which is what the original search used; pass
            ``WIDE_NUMERIC_COLUMNS`` to sweep every numeric feature. Widening multiplies the
            candidate count and therefore the correction burden, so it is opt-in rather than
            the default.
    """
    candidates: dict[str, dict[str, str]] = {}

    for field in BINARY_COLUMNS:
        partition = {
            accession: ("yes" if str(values.get(field, "")).strip().lower() in {"true", "1", "yes"} else "no")
            for accession, values in features.items()
            if field in values
        }
        if _usable(partition):
            candidates[field] = partition

    # J-domain position: the incumbent rule system's most-used single feature.
    position = {
        accession: str(values["j_domain_position"]).strip()
        for accession, values in features.items()
        if str(values.get("j_domain_position", "")).strip()
    }
    if _usable(position):
        candidates["j_domain_position"] = position

    # Structured-domain count, capped so the tail does not become singleton groups.
    counts: dict[str, str] = {}
    for accession, values in features.items():
        raw = str(values.get("n_structured_domains", "")).strip()
        if raw.replace(".", "", 1).isdigit():
            number = int(float(raw))
            counts[accession] = f"{min(number, 4)}+" if number >= 4 else str(number)  # noqa: PLR2004
    if _usable(counts):
        candidates["n_structured_domains"] = counts

    for column, label in numeric_columns or (("idr_fraction", "idr"),):
        candidates.update(numeric_tertiles(features, column, label))

    # Cross the base features systematically. Done after the singles are built, so the
    # combinations are products of exactly the partitions that passed on their own.
    candidates.update(_systematic_combinations(dict(candidates)))

    candidates.update(_architecture_groups(features))

    logger.info("built %s candidate partitions", len(candidates))
    return candidates


def _is_number(value: object) -> bool:
    """Whether a field can be read as a float."""
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _size_matched_null(
    partition: Mapping[str, str],
    readout: Mapping[str, str],
    accessions: Sequence[str],
    rng: random.Random,
) -> float:
    """Cramer's V for a random partition with the same group sizes."""
    labels = [partition[a] for a in accessions]
    rng.shuffle(labels)
    return cramers_v(list(zip(labels, [readout[a] for a in accessions], strict=True)))


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """How one candidate grouping scored against one readout."""

    name: str
    readout: str
    n_proteins: int
    n_groups: int
    candidate_v: float
    reference_v: float
    null_mean_v: float
    p_value: float
    candidate_bits: float = 0.0
    """Mutual information with the readout, in bits, above the candidate's own null."""
    reference_bits: float = 0.0
    p_adjusted: float = 1.0

    @property
    def beats_reference(self) -> bool:
        """Whether it explains the readout enough better than A/B/C to matter.

        A bare inequality is not enough once the sample is large: the permutation gate
        passes everything, so a margin of a fraction of a percent would confirm.
        """
        if self.reference_v <= 0:
            return self.candidate_v > 0
        return (self.candidate_v - self.reference_v) / self.reference_v >= MIN_RELATIVE_MARGIN

    def beats_reference_by(self, criterion: str) -> bool:
        """Whether it beats the reference under the named criterion."""
        if criterion == CRITERION_INFORMATION:
            return self.beats_reference_on_information
        return self.beats_reference

    @property
    def beats_reference_on_information(self) -> bool:
        """Whether it carries more information about the readout than A/B/C does.

        Reported beside the Cramer's V verdict because the two answer different questions
        and can disagree. V asks how close an association is to the maximum its table shape
        permits, and so penalises a finer partition even when the extra groups add real
        information. Mutual information asks only how much the partition tells you about the
        readout, which is the question a search over feature *combinations* needs: a
        combination is always finer than its parents, and information is monotone under
        refinement, so it cannot lose on this measure for being fine alone.
        """
        if self.reference_bits <= 0:
            return self.candidate_bits > 0
        return (self.candidate_bits - self.reference_bits) / self.reference_bits >= MIN_RELATIVE_MARGIN

    @property
    def beats_null(self) -> bool:
        """Whether its content, not its granularity, is doing the work."""
        return self.p_adjusted < SIGNIFICANCE_LEVEL

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "candidate": self.name,
            "readout": self.readout,
            "n_proteins": self.n_proteins,
            "n_groups": self.n_groups,
            "candidate_cramers_v": round(self.candidate_v, 4),
            "reference_cramers_v": round(self.reference_v, 4),
            "improvement_over_reference": round(self.candidate_v - self.reference_v, 4),
            "relative_margin_v": round(
                (self.candidate_v - self.reference_v) / self.reference_v if self.reference_v else 0.0, 4
            ),
            "relative_margin_bits": round(
                (self.candidate_bits - self.reference_bits) / self.reference_bits if self.reference_bits else 0.0, 4
            ),
            "null_mean_cramers_v": round(self.null_mean_v, 4),
            "candidate_bits": round(self.candidate_bits, 5),
            "reference_bits": round(self.reference_bits, 5),
            "information_gain_over_reference": round(self.candidate_bits - self.reference_bits, 5),
            "beats_reference_on_information": self.beats_reference_on_information,
            "p_value": round(self.p_value, 5),
            "p_adjusted": round(self.p_adjusted, 5),
            "beats_reference": self.beats_reference,
            "beats_null": self.beats_null,
        }


def _information_above_null(
    labels: Sequence[str],
    readout: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """Mutual information in bits, above a partition of the same shape assigned at random.

    The null has to be shape-matched, and the reason is measurable. Mutual information rises
    with the number of groups whether or not the extra groups mean anything, so comparing
    raw estimates hands the win to whichever partition is finer. On this dataset the rank
    correlation between a candidate's group count and its raw information advantage over
    A/B/C is +0.76 - the criterion was largely measuring granularity.

    Shuffling the *labels* while keeping their group sizes gives a partition with exactly the
    candidate's granularity and none of its content, so subtracting it leaves what the
    grouping knows that its shape alone does not. This is the same control the Cramer's V
    branch has always used; it was the information branch that lacked it.

    Cramer's V has the mirror-image problem - it penalises fineness - so neither statistic
    can be read raw. Both are reported against this null instead.
    """
    observed = mutual_information(labels, readout)
    rng = random.Random(seed)  # noqa: S311 - a permutation null, not a secret
    shuffled = list(labels)
    draws = []
    # Fewer draws than the V null: this estimates a bias floor, not a p-value, and each draw
    # costs a full contingency table over every protein.
    for _ in range(max(1, n_permutations // 10)):
        rng.shuffle(shuffled)
        draws.append(mutual_information(shuffled, readout))
    return max(0.0, observed - sum(draws) / len(draws))


def score_candidate(  # noqa: PLR0913 - a scored comparison needs both labellings, the readout, two names and the null's settings
    partition: Mapping[str, str],
    reference: Mapping[str, str],
    readout: Mapping[str, str],
    *,
    name: str,
    readout_name: str,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> CandidateResult | None:
    """Score one candidate against one readout, with a size-matched null."""
    shared = sorted(set(partition) & set(reference) & set(readout))
    if len(shared) < MIN_PROTEINS:
        return None

    candidate_v = cramers_v([(partition[a], readout[a]) for a in shared])
    reference_v = cramers_v([(reference[a], readout[a]) for a in shared])

    # Information carried about the readout, as the granularity-fair companion to V. Both
    # sides are measured against their own permutation null, because a finer partition has
    # the larger finite-sample bias and comparing raw estimates would hand it the win.
    readout_values = [readout[a] for a in shared]
    candidate_bits = _information_above_null(
        [partition[a] for a in shared], readout_values, n_permutations=n_permutations, seed=seed
    )
    reference_bits = _information_above_null(
        [reference[a] for a in shared], readout_values, n_permutations=n_permutations, seed=seed
    )

    rng = random.Random(seed)  # noqa: S311 - a permutation null, not a secret
    null = [_size_matched_null(partition, readout, shared, rng) for _ in range(n_permutations)]
    at_least = sum(1 for value in null if value >= candidate_v)
    return CandidateResult(
        name=name,
        readout=readout_name,
        n_proteins=len(shared),
        n_groups=len({partition[a] for a in shared}),
        candidate_v=candidate_v,
        reference_v=reference_v,
        null_mean_v=sum(null) / len(null) if null else 0.0,
        candidate_bits=candidate_bits,
        reference_bits=reference_bits,
        p_value=(at_least + 1) / (n_permutations + 1),
    )


def search_partitions(  # noqa: PLR0913 - a search needs candidates, a reference, and both readout sets
    candidates: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, str],
    discovery: Mapping[str, str],
    validation: Mapping[str, Mapping[str, str]],
    *,
    discovery_name: str = "discovery",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    criterion: str = CRITERION_V,
) -> dict[str, object]:
    """Rank candidates on one readout, then test the survivors on readouts held back.

    A candidate is only reported as beating A/B/C if it does so on a readout it was never
    ranked against. Selecting on one and confirming on another is what separates a grouping
    that carries biology from one that fits the noise in whichever readout was used to
    find it.
    """
    ranked = [
        result
        for name, partition in sorted(candidates.items())
        if (
            result := score_candidate(
                partition,
                reference,
                discovery,
                name=name,
                readout_name=discovery_name,
                n_permutations=n_permutations,
                seed=seed,
            )
        )
        is not None
    ]
    adjusted = benjamini_hochberg({item.name: item.p_value for item in ranked})
    ranked = [
        CandidateResult(
            name=item.name,
            readout=item.readout,
            n_proteins=item.n_proteins,
            n_groups=item.n_groups,
            candidate_v=item.candidate_v,
            reference_v=item.reference_v,
            null_mean_v=item.null_mean_v,
            candidate_bits=item.candidate_bits,
            reference_bits=item.reference_bits,
            p_value=item.p_value,
            p_adjusted=adjusted[item.name],
        )
        for item in ranked
    ]
    if criterion == CRITERION_INFORMATION:
        ranked.sort(key=lambda item: -item.candidate_bits)
    else:
        ranked.sort(key=lambda item: -item.candidate_v)

    survivors = [item for item in ranked if item.beats_reference_by(criterion) and item.beats_null]

    scored: list[CandidateResult] = []
    for survivor in survivors:
        for readout_name, readout in sorted(validation.items()):
            result = score_candidate(
                candidates[survivor.name],
                reference,
                readout,
                name=survivor.name,
                readout_name=readout_name,
                n_permutations=n_permutations,
                seed=seed,
            )
            if result is not None:
                scored.append(result)

    # The confirmations are their own family of tests and are corrected as one. Left at
    # the dataclass default of 1.0 - which is what happened first - `beats_null` was
    # false for every one of them, so a candidate scoring a perfect 1.000 against A/B/C's
    # 0.000 on a held-out readout still reported as unconfirmed. The search returned no
    # winners regardless of the data, which looks exactly like a careful negative result.
    confirm_adjusted = benjamini_hochberg(
        {f"{item.name}|{item.readout}": item.p_value for item in scored},
    )
    confirmations = [
        CandidateResult(
            name=item.name,
            readout=item.readout,
            n_proteins=item.n_proteins,
            n_groups=item.n_groups,
            candidate_v=item.candidate_v,
            reference_v=item.reference_v,
            null_mean_v=item.null_mean_v,
            candidate_bits=item.candidate_bits,
            reference_bits=item.reference_bits,
            p_value=item.p_value,
            p_adjusted=confirm_adjusted[f"{item.name}|{item.readout}"],
        ).to_json_dict()
        for item in scored
    ]

    beats_key = "beats_reference_on_information" if criterion == CRITERION_INFORMATION else "beats_reference"
    confirmed = {item["candidate"] for item in confirmations if item[beats_key] and item["beats_null"]}

    return {
        "n_candidates_tested": len(ranked),
        "criterion": criterion,
        "discovery_readout": discovery_name,
        "validation_readouts": sorted(validation),
        "n_beating_reference_on_discovery": len(survivors),
        "n_confirmed_on_held_out_readout": len(confirmed),
        "confirmed_candidates": sorted(confirmed),
        "discovery": [item.to_json_dict() for item in ranked],
        "validation": confirmations,
        "interpretation": (
            "Candidates are ranked on the discovery readout and then re-tested on readouts "
            "held back from ranking. Correction is across every candidate tested, not the "
            "promising ones, and each is compared against a random partition with the same "
            "group sizes - a five-group split beats a three-group split at almost anything, "
            "so granularity has to be controlled before content can be claimed. A candidate "
            "listed as confirmed beat A/B/C on a readout it was never selected against, "
            f"judged by the '{criterion}' criterion at both stages."
        ),
    }
