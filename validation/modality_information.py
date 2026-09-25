"""Whether grammar, structure, and function carry the same information or different information.

The project has three descriptions of every protein. **Grammar** is what the layout pipeline
reads off the sequence: how the domains are arranged, how much of the chain is disordered,
how the charge is distributed. **Structure** is what AlphaFold and the Poisson-Boltzmann
solver report about the folded model. **Function** is the annotation - GO terms and keywords,
observed or transferred by homology.

That the three correlate is not in question and would not be interesting: everything in
biology correlates at n = 10^5. The question worth asking is whether grammar tells you
anything about function that structure does not *already* tell you. Three answers are
possible and they mean different things:

- **Redundant.** Grammar and structure carry overlapping information; either one alone gets
  you most of the way, and the second adds little. The modalities are tied, and a classifier
  needs only one of them.
- **Independent.** Each carries its own information and they simply add. Using both is worth
  it, but neither explains the other.
- **Synergistic.** The pair together says more than the sum of its parts - a combination of
  grammar and structure identifies a function that neither identifies alone. This is the case
  that would justify the dual-layer design on information-theoretic grounds rather than
  intuition.

Method
------
For a functional term *Y*, with grammar profile *G* and structure profile *S*::

    redundancy = I(G;Y) + I(S;Y) - I(G,S;Y)

positive for overlap, negative for synergy, near zero for independence. This is the
interaction information, and its sign is the whole result.

Two corrections are not optional here. Mutual information is biased *upward* by finite
sampling - two independent variables show positive MI simply because cells are unevenly
filled - so every estimate carries a Miller-Madow correction and, more importantly, a
permutation null obtained by shuffling *Y* alone. An MI that does not exceed its own null is
reported as zero information, not as a small effect.

The second concerns provenance. 74.5% of the functional terms in this dataset were
transferred by homology rather than observed, and ``transfer_annotations`` states the
contract plainly: a conclusion that holds only on transferred terms is a conclusion about
the transfer method. So the analysis runs on both sets and reports them separately; the
observed-only result is the one that carries weight, and the transferred one is a check on
whether the transfer preserved the relationship or manufactured it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validation.recurrence import benjamini_hochberg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Terms rarer than this cannot support a contingency table against two profiles.
MIN_TERM_COUNT = 50

# Proteins needed before a three-way table is worth estimating at all.
MIN_PROTEINS = 200

# Shuffles used to establish the bias floor for each mutual information estimate.
DEFAULT_PERMUTATIONS = 200

# Quantile bins per continuous feature. Three keeps the joint state space small enough that
# cells stay populated: two features per modality gives 9 states, and the pair gives 81.
DEFAULT_BINS = 3

SIGNIFICANCE_LEVEL = 0.05

# Interaction information is reported in bits; below this the sign is not worth naming.
NEGLIGIBLE_BITS = 0.001

# Above this fractional gain, combining the modalities is worth it even when they overlap.
SUBSTANTIAL_GAIN = 0.1

# Provenances whose labels were produced from sequence, which gives the sequence-derived
# grammar an inside track on them. Any result on these carries the caveat automatically, so
# that renaming a tier in a driver cannot quietly drop it.
SEQUENCE_DERIVED_PROVENANCES = frozenset({"transferred", "homology_transfer", "electronic"})


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy in bits, from raw counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            out -= p * math.log2(p)
    return out


def _miller_madow(observed_cells: int, total: int) -> float:
    """Bias correction for a plug-in entropy estimate.

    The plug-in estimator underestimates entropy by roughly (k-1)/(2N ln 2), where k is the
    number of states actually seen. Applied to every entropy term, this removes most of the
    spurious mutual information that finite sampling produces between independent variables.
    """
    if total <= 0:
        return 0.0
    return (observed_cells - 1) / (2 * total * math.log(2))


def mutual_information(first: Sequence[object], second: Sequence[object]) -> float:
    """Mutual information in bits, Miller-Madow corrected, floored at zero.

    Negative values are an artefact of the correction over-shooting on sparse tables; they
    are clamped, because a negative mutual information is not a meaningful quantity.
    """
    if len(first) != len(second) or not first:
        return 0.0
    total = len(first)
    joint = Counter(zip(first, second, strict=True))
    left = Counter(first)
    right = Counter(second)

    h_left = _entropy(list(left.values())) + _miller_madow(len(left), total)
    h_right = _entropy(list(right.values())) + _miller_madow(len(right), total)
    h_joint = _entropy(list(joint.values())) + _miller_madow(len(joint), total)
    return max(0.0, h_left + h_right - h_joint)


def quantile_bins(values: Sequence[float], n_bins: int = DEFAULT_BINS) -> list[int]:
    """Assign each value to a quantile bin.

    Quantiles rather than equal-width bins because these features are heavily skewed - most
    proteins have few domains and low disorder - and equal-width binning would put almost
    every protein in one bin, destroying the information being measured.
    """
    if not values:
        return []
    ordered = sorted(values)
    cuts = [ordered[min(len(ordered) - 1, int(len(ordered) * i / n_bins))] for i in range(1, n_bins)]
    return [sum(1 for cut in cuts if value > cut) for value in values]


def profile_states(
    accessions: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
    *,
    n_bins: int = DEFAULT_BINS,
) -> list[tuple[int, ...]]:
    """Reduce several continuous features to one discrete state per protein.

    Each feature is binned separately and the bins are tupled, so a profile keeps which
    feature was high rather than collapsing them into a score that could reach the same
    value two different ways.
    """
    columns = [quantile_bins([features[a].get(name, 0.0) for a in accessions], n_bins) for name in names]
    return [tuple(column[index] for column in columns) for index in range(len(accessions))]


@dataclass(frozen=True)
class Modality:
    """One description of a set of proteins, and which of its features form the profile.

    Features and the names drawn from them travel together because they are only meaningful
    as a pair: the same table can define a coarse profile or a fine one, and which was used
    has to be recorded with any result computed from it.
    """

    features: Mapping[str, Mapping[str, float]]
    names: Sequence[str]

    def states(self, accessions: Sequence[str], *, n_bins: int = DEFAULT_BINS) -> list[tuple[int, ...]]:
        """Discrete profile per accession."""
        return profile_states(accessions, self.features, self.names, n_bins=n_bins)


@dataclass(frozen=True)
class ReportSettings:
    """How many terms to test, how hard to test them, and where the terms came from."""

    max_terms: int = 40
    n_permutations: int = DEFAULT_PERMUTATIONS
    provenance: str = "unspecified"


DEFAULT_SETTINGS = ReportSettings()


@dataclass(frozen=True)
class TermResult:
    """Information the two modalities carry about one functional term."""

    term: str
    n_proteins: int
    n_with_term: int
    grammar_bits: float
    structure_bits: float
    joint_bits: float
    interaction_bits: float
    """I(G;Y) + I(S;Y) - I(G,S;Y). Positive is redundant, negative is synergistic."""
    grammar_null: float
    structure_null: float
    joint_null: float
    p_value: float
    p_adjusted: float = 1.0

    @property
    def relationship(self) -> str:
        """Whether the modalities overlap, complement, or neither, on this term.

        The emptiness check is on the *joint* information, not on the two modalities
        separately. Testing the parts would discard exactly the case worth finding: under
        pure synergy neither modality alone says anything about the term, and only their
        combination does, so a guard reading the parts reports the strongest possible
        result as "nothing here".
        """
        if self.joint_bits <= self.joint_null:
            return "neither_informative"
        if abs(self.interaction_bits) < NEGLIGIBLE_BITS:
            return "independent"
        return "redundant" if self.interaction_bits > 0 else "synergistic"

    @property
    def redundancy_fraction(self) -> float:
        """What share of the weaker modality's information the two hold in common.

        The sign of the interaction says *whether* the modalities overlap; this says how
        much, and the two can point different ways. Every term can read as "redundant" while
        the joint profile still carries far more than either modality alone - a very
        different conclusion from "either one will do", and one the sign alone would hide.
        """
        weaker = min(
            max(0.0, self.grammar_bits - self.grammar_null),
            max(0.0, self.structure_bits - self.structure_null),
        )
        if weaker <= 0:
            return 0.0
        return self.interaction_bits / weaker

    @property
    def joint_gain(self) -> float:
        """How much the pair adds over the better single modality, as a fraction of it."""
        best = max(0.0, self.grammar_bits - self.grammar_null, self.structure_bits - self.structure_null)
        if best <= 0:
            return 0.0
        return (max(0.0, self.joint_bits - self.joint_null) - best) / best

    @property
    def significant(self) -> bool:
        """Whether the joint information exceeds what shuffled labels produce."""
        return self.p_adjusted < SIGNIFICANCE_LEVEL

    def to_json_dict(self) -> dict[str, object]:
        """Render for the JSON report."""
        return {
            "term": self.term,
            "n_proteins": self.n_proteins,
            "n_with_term": self.n_with_term,
            "grammar_bits": round(self.grammar_bits, 5),
            "structure_bits": round(self.structure_bits, 5),
            "joint_bits": round(self.joint_bits, 5),
            "interaction_bits": round(self.interaction_bits, 5),
            "grammar_null_bits": round(self.grammar_null, 5),
            "structure_null_bits": round(self.structure_null, 5),
            "joint_null_bits": round(self.joint_null, 5),
            "relationship": self.relationship,
            "redundancy_fraction": round(self.redundancy_fraction, 4),
            "joint_gain_over_best_single": round(self.joint_gain, 4),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "significant": self.significant,
        }


def _shuffled(values: Sequence[object], seed: int) -> list[object]:
    """Deterministic Fisher-Yates shuffle, so a report is reproducible from its inputs."""
    out = list(values)
    state = seed or 1
    for index in range(len(out) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (1 << 31)
        swap = state % (index + 1)
        out[index], out[swap] = out[swap], out[index]
    return out


def analyse_term(
    term: str,
    labels: Sequence[int],
    grammar: Sequence[tuple[int, ...]],
    structure: Sequence[tuple[int, ...]],
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> TermResult:
    """Measure what each modality, and their combination, says about one term."""
    joint_profile = [(g, s) for g, s in zip(grammar, structure, strict=True)]
    grammar_bits = mutual_information(grammar, labels)
    structure_bits = mutual_information(structure, labels)
    joint_bits = mutual_information(joint_profile, labels)

    # The null shuffles the labels, which destroys any relationship to either modality while
    # preserving how common the term is and how the profiles are distributed. Whatever
    # information survives that is the estimator's own bias.
    grammar_null_draws, structure_null_draws, joint_null_draws = [], [], []
    for index in range(n_permutations):
        shuffled = _shuffled(labels, seed=index + 1)
        grammar_null_draws.append(mutual_information(grammar, shuffled))
        structure_null_draws.append(mutual_information(structure, shuffled))
        joint_null_draws.append(mutual_information(joint_profile, shuffled))

    exceeded = sum(1 for draw in joint_null_draws if draw >= joint_bits)
    p_value = (exceeded + 1) / (n_permutations + 1)

    grammar_null = sum(grammar_null_draws) / max(1, len(grammar_null_draws))
    structure_null = sum(structure_null_draws) / max(1, len(structure_null_draws))
    joint_null = sum(joint_null_draws) / max(1, len(joint_null_draws))

    # Subtract each estimate's own bias floor before combining them, so the interaction term
    # is not dominated by the fact that the joint table has the most cells and therefore the
    # largest bias.
    interaction = (
        max(0.0, grammar_bits - grammar_null)
        + max(0.0, structure_bits - structure_null)
        - max(0.0, joint_bits - joint_null)
    )

    return TermResult(
        term=term,
        n_proteins=len(labels),
        n_with_term=sum(labels),
        grammar_bits=grammar_bits,
        structure_bits=structure_bits,
        joint_bits=joint_bits,
        interaction_bits=interaction,
        grammar_null=grammar_null,
        structure_null=structure_null,
        joint_null=joint_null,
        p_value=p_value,
    )


def modality_report(
    terms_by_accession: Mapping[str, Sequence[str]],
    grammar: Modality,
    structure: Modality,
    *,
    settings: ReportSettings = DEFAULT_SETTINGS,
) -> dict[str, object]:
    """Ask, for the commonest functional terms, how grammar and structure divide the work.

    Args:
        terms_by_accession: Functional terms per protein.
        grammar: Sequence-derived features and the profile drawn from them.
        structure: Model-derived features and the profile drawn from them.
        settings: Term count, permutation count, and the provenance of the terms -
            ``observed`` or ``transferred``, recorded rather than inferred, because the two
            must not be read as the same kind of evidence.

    Returns:
        Per-term results plus a summary counting how many terms fall in each relationship.
    """
    max_terms, n_permutations = settings.max_terms, settings.n_permutations
    provenance = settings.provenance
    shared = sorted(
        a for a in terms_by_accession if a in grammar.features and a in structure.features and terms_by_accession[a]
    )
    if len(shared) < MIN_PROTEINS:
        return {
            "provenance": provenance,
            "n_proteins": len(shared),
            "results": [],
            "summary": {},
            "interpretation": (
                f"Only {len(shared)} proteins carry a term and both feature sets; "
                f"{MIN_PROTEINS} are needed before a three-way table means anything."
            ),
        }

    grammar_states = grammar.states(shared)
    structure_states = structure.states(shared)

    counts = Counter(term for a in shared for term in set(terms_by_accession[a]))
    chosen = [term for term, count in counts.most_common(max_terms) if count >= MIN_TERM_COUNT]

    results = []
    for term in chosen:
        labels = [1 if term in set(terms_by_accession[a]) else 0 for a in shared]
        # A term carried by every protein has no variance to explain.
        if 0 < sum(labels) < len(labels):
            results.append(analyse_term(term, labels, grammar_states, structure_states, n_permutations=n_permutations))

    adjusted = benjamini_hochberg({r.term: r.p_value for r in results})
    results = [
        TermResult(
            term=r.term,
            n_proteins=r.n_proteins,
            n_with_term=r.n_with_term,
            grammar_bits=r.grammar_bits,
            structure_bits=r.structure_bits,
            joint_bits=r.joint_bits,
            interaction_bits=r.interaction_bits,
            grammar_null=r.grammar_null,
            structure_null=r.structure_null,
            joint_null=r.joint_null,
            p_value=r.p_value,
            p_adjusted=adjusted.get(r.term, 1.0),
        )
        for r in results
    ]

    informative = [r for r in results if r.significant and r.relationship != "neither_informative"]
    summary = Counter(r.relationship for r in informative)

    if not informative:
        note = "No term carries information from either modality beyond what shuffled labels produce."
    else:
        dominant = summary.most_common(1)[0][0]
        share = summary[dominant] / len(informative)
        overlaps = sorted(r.redundancy_fraction for r in informative)
        gains = sorted(r.joint_gain for r in informative)
        median_overlap = overlaps[len(overlaps) // 2]
        median_gain = gains[len(gains) // 2]
        note = f"Of {len(informative)} informative terms, {summary[dominant]} ({share:.0%}) are {dominant}. "
        if dominant == "redundant":
            # The sign alone would justify dropping a layer; the magnitude usually does not.
            # State both, rather than letting the label carry a conclusion it cannot support.
            note += (
                f"They overlap by a median {median_overlap:.0%} of the weaker modality's "
                f"information, but the pair still carries a median {median_gain:.0%} more than "
                f"the better single modality alone: the overlap is partial, and each layer "
                f"holds function-relevant information the other does not."
                if median_gain > SUBSTANTIAL_GAIN
                else (
                    f"They overlap by a median {median_overlap:.0%}, and the pair adds only "
                    f"{median_gain:.0%} over the better single modality - one layer would do."
                )
            )
        elif dominant == "synergistic":
            note += (
                "Grammar and structure are complementary: their combination identifies "
                "functions that neither identifies alone, which is the information-theoretic "
                "case for keeping both layers."
            )
        else:
            note += "The two modalities carry separate, additive information about function."
        if provenance in SEQUENCE_DERIVED_PROVENANCES:
            # The transfer used profile HMMs over *sequence*, so grammar - also sequence
            # derived - has an inside track on these labels. Structure took no part in it,
            # which is what makes the information structure carries here worth something.
            note += (
                " These terms were transferred by sequence homology, which gives the "
                "sequence-derived grammar an inside track on them; structure took no part in "
                "the transfer, so what it carries here cannot be an artefact of the method."
            )

    return {
        "provenance": provenance,
        "n_proteins": len(shared),
        "n_terms_tested": len(results),
        "grammar_profile": list(grammar.names),
        "structure_profile": list(structure.names),
        "n_grammar_states": len(set(grammar_states)),
        "n_structure_states": len(set(structure_states)),
        "results": [r.to_json_dict() for r in results],
        "summary": dict(summary),
        "median_redundancy_fraction": (
            round(sorted(r.redundancy_fraction for r in informative)[len(informative) // 2], 4) if informative else 0.0
        ),
        "median_joint_gain": (
            round(sorted(r.joint_gain for r in informative)[len(informative) // 2], 4) if informative else 0.0
        ),
        "interpretation": note,
    }
