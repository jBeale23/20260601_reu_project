"""Compositional grammar of a sequence: n-mer syntax, entropy, and complexity.

The alignment-free layer (:mod:`domain_layout.shark`) asks "how similar is this region to
a reference region?". This module asks a different and complementary question: "how is
this region *built*, regardless of what it resembles?" That matters because the regions
routed away from the MSA - IDRs, G/F-rich stretches, linkers, termini - are precisely the
ones where homology is undetectable but composition is highly structured.

Three measures, in increasing order of what they assume:

**Reduced-alphabet n-mer syntax.** Twenty amino acids give 20^k k-mers, which is far too
sparse to estimate a distribution over at k > 2. Projecting onto a chemical alphabet
(:data:`REDUCED_ALPHABETS`) collapses substitutions that preserve biophysics, so a G/F-rich
block in one organism and its diverged counterpart in another produce the same grammar
even when no aligner would relate them.

**Shannon entropy, and the entropy rate.** Entropy over 1-mers is plain composition bias.
The interesting quantity is the *entropy rate* ``h_k = H(k) - H(k-1)``: how much a symbol
still surprises once the preceding ``k-1`` are known. A low ``h_2`` against a high ``h_1``
means order carries structure that composition alone does not - a repeat grammar. The
profile of ``h_k`` over increasing ``k`` is the hierarchy: each level reports what the
level below could not explain.

**Kolmogorov complexity, approximately.** True Kolmogorov complexity is uncomputable, so
this uses the standard practical estimator: the length of the sequence after lossless
compression, which is an *upper bound* on ``K``. It is reported as
:attr:`GrammarProfile.compression_ratio` and never called Kolmogorov complexity outright.
Compression catches long-range and non-local repetition that a fixed-``k`` entropy cannot
see - a 40-residue motif recurring twice in a 400-residue IDR barely moves ``h_3`` but
compresses noticeably.

Estimator bias is the trap here. Plug-in entropy from a short sequence is biased
*downward*, so a 30-residue linker looks more ordered than a 300-residue one purely
through sample size. Every entropy returned by this module carries the Miller-Madow
correction, and :attr:`GrammarProfile.n_observations` is reported so a caller can discount
a measurement taken from too little sequence.
"""

from __future__ import annotations

import lzma
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Chemical reductions of the twenty standard residues. Each collapses substitutions that
# conserve the property the grammar is meant to track, which is what lets a diverged
# homolog produce the same n-mer syntax as its counterpart.
_CHEMICAL7 = {
    **dict.fromkeys("AVLIM", "H"),  # aliphatic / hydrophobic
    **dict.fromkeys("FWY", "R"),  # aromatic
    **dict.fromkeys("STNQ", "P"),  # polar uncharged
    **dict.fromkeys("KRH", "+"),  # positive
    **dict.fromkeys("DE", "-"),  # negative
    **dict.fromkeys("GP", "T"),  # turn / flexibility drivers
    **dict.fromkeys("C", "C"),  # cysteine: disulfides make it its own class
}
_HYDROPHOBIC2 = {
    **dict.fromkeys("AVLIMFWYC", "H"),
    **dict.fromkeys("GPSTNQDEKRH", "P"),
}
_CHARGE3 = {
    **dict.fromkeys("KRH", "+"),
    **dict.fromkeys("DE", "-"),
    **dict.fromkeys("ACFGILMNPQSTVWY", "0"),
}

REDUCED_ALPHABETS: dict[str, Mapping[str, str]] = {
    "chemical7": _CHEMICAL7,
    "hydrophobic2": _HYDROPHOBIC2,
    "charge3": _CHARGE3,
}
DEFAULT_ALPHABET = "chemical7"

# Symbol for residues outside the twenty standard ones (X, U, B, Z). Kept as its own class
# rather than dropped, so a run of unknowns does not silently masquerade as a repeat.
UNKNOWN_SYMBOL = "?"

# Highest n-mer order profiled by default. Above this the counts are too sparse for the
# entropy estimate to mean anything on a typical few-hundred-residue region, even after
# the alphabet reduction.
DEFAULT_MAX_ORDER = 4

# Below this many residues an entropy estimate is dominated by its own bias. Profiles are
# still produced - a caller may legitimately want the composition - but flagged.
MIN_RELIABLE_LENGTH = 25

# Shuffles used for the composition-matched control. Five is enough to place the mean; the
# control is a reference point, not a hypothesis test.
DEFAULT_N_SHUFFLES = 5

# LZMA settings, shared by the observation and its shuffle control so the ratio between
# them stays valid. Preset 6 rather than 9|PRESET_EXTREME: on strings of a few hundred
# residues the two find the same repeats and return the same order scores to three decimal
# places, but the extreme preset costs 193 ms per region against 5.6 ms - a 34x difference
# that decides whether profiling a proteome takes minutes or hours.
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "preset": 6}]

# Order score above which a region reads as genuinely repetitive. Calibrated against
# synthetic controls at the lengths this pipeline actually sees: a composition-matched
# shuffle scores ~0, a G/F-like stretch with no periodicity ~0.05-0.15, and a true tandem
# repeat >0.6. See docs/grammar_calibration.md.
LOW_COMPLEXITY_ORDER_SCORE = 0.35

# Normalized first-order entropy below which a region is built from too few residue
# classes to carry sequence information - a homopolymer scores 0.0, and a two-class
# alternating stretch about 0.33 on the seven-class alphabet.
DEGENERATE_ENTROPY = 0.5

_LOG2 = math.log(2.0)


def reduce_sequence(sequence: str, *, alphabet: str = DEFAULT_ALPHABET) -> str:
    """Project a sequence onto a reduced chemical alphabet.

    Raises:
        KeyError: If ``alphabet`` is not one of :data:`REDUCED_ALPHABETS`.
    """
    table = REDUCED_ALPHABETS[alphabet]
    return "".join(table.get(residue, UNKNOWN_SYMBOL) for residue in sequence.upper())


def nmer_counts(sequence: str, order: int) -> Counter[str]:
    """Count overlapping n-mers of length ``order``."""
    if order <= 0 or len(sequence) < order:
        return Counter()
    return Counter(sequence[index : index + order] for index in range(len(sequence) - order + 1))


def shannon_entropy(counts: Mapping[str, int] | Counter[str], *, corrected: bool = True) -> float:
    """Shannon entropy of a symbol distribution, in bits.

    Args:
        counts: Observed counts per symbol.
        corrected: Apply the Miller-Madow bias correction. Plug-in entropy systematically
            under-estimates when symbols outnumber observations, which is the normal
            regime for n-mers of a short region; leaving it off makes short sequences look
            spuriously ordered.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)

    if corrected:
        observed_symbols = sum(1 for count in counts.values() if count > 0)
        entropy += (observed_symbols - 1) / (2.0 * total * _LOG2)
    return entropy


def block_entropies(sequence: str, *, max_order: int = DEFAULT_MAX_ORDER) -> list[float]:
    """Entropy ``H(k)`` of the n-mer distribution for ``k = 1 .. max_order``, in bits."""
    return [shannon_entropy(nmer_counts(sequence, order)) for order in range(1, max(1, max_order) + 1)]


def entropy_rates(sequence: str, *, max_order: int = DEFAULT_MAX_ORDER) -> list[float]:
    """Conditional entropy rates ``h_k = H(k) - H(k-1)``, in bits per residue.

    This is the hierarchical part of the grammar. ``h_1`` is composition alone; each
    further ``h_k`` reports only the surprise that the shorter context failed to explain.
    A repeat grammar shows a steep fall from ``h_1`` to ``h_2``; a compositionally biased
    but otherwise disordered stretch shows a low ``h_1`` and a flat tail.

    Rates are clamped at zero: entropy is non-increasing in context length in the limit,
    and a negative value is an artefact of estimating both terms from finite sequence.
    """
    blocks = block_entropies(sequence, max_order=max_order)
    rates: list[float] = []
    previous = 0.0
    for entropy in blocks:
        rates.append(max(0.0, entropy - previous))
        previous = entropy
    return rates


def compression_ratio(sequence: str) -> float:
    """Compressed size over raw size: a computable upper bound on Kolmogorov complexity.

    Kolmogorov complexity itself is uncomputable, so this substitutes the length of a
    lossless encoding, which bounds it from above. Values near 1 mean the compressor found
    no structure; low values mean strong repetition. LZMA is used rather than zlib for its
    much larger window, so repeats separated by hundreds of residues are still detected.

    Returns 1.0 for the empty sequence: nothing to compress is not evidence of structure.
    """
    if not sequence:
        return 1.0
    raw = sequence.encode("ascii", errors="replace")
    # Container headers would dominate on short regions, so the raw LZMA1 stream is used
    # with no CRC or end marker.
    compressed = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    return len(compressed) / len(raw)


def order_score(
    sequence: str,
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = 0,
) -> float:
    """How much of a sequence's compressibility comes from *order* rather than composition.

    Raw compression ratio cannot be thresholded, because it depends heavily on length: a
    uniformly random 25-residue string compresses to 1.32 of its size while a random
    1500-residue one reaches 0.515. Comparing two regions of different length on that
    number compares their lengths, not their grammars.

    The fix is the standard shuffle control. Shuffling a sequence destroys all order while
    preserving its length and its exact residue composition, so the ratio of the real
    compressed size to the mean shuffled one isolates the contribution of arrangement:

        order_score = 1 - C(x) / mean(C(shuffle(x)))

    Zero means the sequence is no more compressible than its own composition already
    implies. Values approaching one mean strong repetition. Negative values are clamped -
    they arise from compressor noise, not from a sequence being less ordered than random.

    One blind spot is structural, not incidental: a homopolymer scores exactly zero,
    because every permutation of ``QQQQ...`` is the same string, so the control equals the
    observation. That is the case first-order entropy detects perfectly (``H = 0``), which
    is why :attr:`GrammarProfile.is_low_complexity` and
    :attr:`GrammarProfile.is_compositionally_degenerate` are separate tests rather than one
    combined score - each covers what the other cannot see.
    """
    reduced_length = len(sequence)
    if reduced_length < MIN_RELIABLE_LENGTH:
        return 0.0

    observed = _compressed_size(sequence)
    if observed <= 0:
        return 0.0

    # Deterministic, seeded shuffling for reproducibility; not a security context.
    rng = random.Random(seed)  # noqa: S311
    symbols = list(sequence)
    total = 0
    for _ in range(max(1, n_shuffles)):
        rng.shuffle(symbols)
        total += _compressed_size("".join(symbols))
    expected = total / max(1, n_shuffles)
    if expected <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - observed / expected))


def _compressed_size(sequence: str) -> int:
    if not sequence:
        return 0
    return len(
        lzma.compress(sequence.encode("ascii", errors="replace"), format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS),
    )


def normalized_compression_distance(first: str, second: str) -> float:
    """Normalized compression distance between two sequences, in [0, ~1].

    ``NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`` (Cilibrasi and Vitanyi).
    Zero means one sequence tells you everything about the other; one means they share no
    exploitable structure. Unlike a SHARK score this needs no substitution matrix and no
    notion of residue similarity, so it stays meaningful on regions whose alphabet
    statistics have drifted far enough that scoring matrices no longer apply.
    """
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0

    size_first = _compressed_size(first)
    size_second = _compressed_size(second)
    size_joint = _compressed_size(first + second)
    denominator = max(size_first, size_second)
    if denominator <= 0:
        return 0.0
    distance = (size_joint - min(size_first, size_second)) / denominator
    # A real compressor is not a perfect one, so the theoretical [0, 1] bound can be
    # exceeded slightly; clamping keeps downstream arithmetic honest.
    return min(1.0, max(0.0, distance))


@dataclass(frozen=True, slots=True)
class GrammarProfile:
    """The compositional grammar of one sequence."""

    length: int
    alphabet: str
    block_entropies: tuple[float, ...]
    entropy_rates: tuple[float, ...]
    compression_ratio: float
    order_score: float
    n_observations: int
    dominant_nmer: str
    dominant_fraction: float

    @property
    def is_reliable(self) -> bool:
        """Whether the sequence is long enough for the entropy terms to mean anything."""
        return self.length >= MIN_RELIABLE_LENGTH

    @property
    def normalized_entropy(self) -> float:
        """First-order entropy as a fraction of the maximum for this alphabet, in [0, 1].

        Comparable across alphabets in a way that raw bits are not.
        """
        symbols = len(set(REDUCED_ALPHABETS[self.alphabet].values())) + 1
        ceiling = math.log2(symbols)
        if ceiling <= 0 or not self.block_entropies:
            return 0.0
        return min(1.0, max(0.0, self.block_entropies[0] / ceiling))

    @property
    def is_compositionally_degenerate(self) -> bool:
        """Whether the region is built from too few distinct residue classes.

        The complement of :attr:`is_low_complexity`. A homopolymer or near-homopolymer is
        invisible to the shuffle control - its shuffles are itself - but has near-zero
        first-order entropy, so this is the test that catches it.
        """
        return self.is_reliable and self.normalized_entropy < DEGENERATE_ENTROPY

    @property
    def is_low_complexity(self) -> bool:
        """Whether this region is repetitive beyond what its composition alone explains.

        Thresholded on :func:`order_score` rather than the raw compression ratio, which is
        far too length-dependent to compare across regions. A compositionally skewed but
        non-repetitive stretch - an ordinary G/F-rich linker - is deliberately *not*
        flagged here; that is what the entropy terms describe.
        """
        return self.is_reliable and self.order_score >= LOW_COMPLEXITY_ORDER_SCORE

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "length": self.length,
            "alphabet": self.alphabet,
            "block_entropies": [round(value, 4) for value in self.block_entropies],
            "entropy_rates": [round(value, 4) for value in self.entropy_rates],
            "normalized_entropy": round(self.normalized_entropy, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "order_score": round(self.order_score, 4),
            "n_observations": self.n_observations,
            "dominant_nmer": self.dominant_nmer,
            "dominant_fraction": round(self.dominant_fraction, 4),
            "is_low_complexity": self.is_low_complexity,
            "is_compositionally_degenerate": self.is_compositionally_degenerate,
            "is_reliable": self.is_reliable,
        }


def profile_sequence(
    sequence: str,
    *,
    alphabet: str = DEFAULT_ALPHABET,
    max_order: int = DEFAULT_MAX_ORDER,
    dominant_order: int = 3,
    seed: int = 0,
) -> GrammarProfile:
    """Measure the compositional grammar of one sequence."""
    reduced = reduce_sequence(sequence, alphabet=alphabet)
    blocks = block_entropies(reduced, max_order=max_order)
    rates = entropy_rates(reduced, max_order=max_order)

    counts = nmer_counts(reduced, min(dominant_order, max(1, len(reduced))))
    total = sum(counts.values())
    dominant, dominant_count = counts.most_common(1)[0] if counts else ("", 0)

    return GrammarProfile(
        length=len(sequence),
        alphabet=alphabet,
        block_entropies=tuple(blocks),
        entropy_rates=tuple(rates),
        # Measured on the reduced string, which is what the entropies describe.
        compression_ratio=compression_ratio(reduced),
        order_score=order_score(reduced, seed=seed),
        n_observations=total,
        dominant_nmer=dominant,
        dominant_fraction=(dominant_count / total) if total else 0.0,
    )


@dataclass(frozen=True, slots=True)
class GrammarAnomaly:
    """How unusual one sequence's grammar is against a fitted corpus background."""

    surprisal_per_residue: float
    z_score: float
    percentile: float

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "surprisal_per_residue": round(self.surprisal_per_residue, 4),
            "z_score": round(self.z_score, 3),
            "percentile": round(self.percentile, 4),
        }


class CorpusGrammar:
    """Background n-mer model fitted over a corpus, used to score anomalous grammar.

    The novelty score elsewhere in this project is architectural: it asks whether a
    protein's *domains* are arranged unusually. This asks whether its unalignable sequence
    is *written* unusually, which is an independent signal - a protein can have a textbook
    architecture and a linker unlike any other in the corpus, or vice versa.

    Scoring is mean surprisal per residue under a Laplace-smoothed n-mer model. Smoothing
    is not optional: an unsmoothed model assigns probability zero to any n-mer it never
    saw, and a single unseen n-mer would send the surprisal of an otherwise ordinary
    region to infinity.
    """

    def __init__(self, *, order: int = 3, alphabet: str = DEFAULT_ALPHABET) -> None:
        """Create an unfitted model of the given n-mer order."""
        self.order = max(1, order)
        self.alphabet = alphabet
        self._counts: Counter[str] = Counter()
        self._total = 0
        self._vocabulary = 0
        self._reference_scores: tuple[float, ...] = ()

    @property
    def is_fitted(self) -> bool:
        """Whether the model has seen any sequence."""
        return self._total > 0

    def fit(self, sequences: Iterable[str]) -> CorpusGrammar:
        """Accumulate background n-mer counts, then calibrate the score distribution."""
        observed: list[str] = []
        for sequence in sequences:
            reduced = reduce_sequence(sequence, alphabet=self.alphabet)
            counts = nmer_counts(reduced, self.order)
            if not counts:
                continue
            self._counts.update(counts)
            observed.append(reduced)

        self._total = sum(self._counts.values())
        symbols = len(set(REDUCED_ALPHABETS[self.alphabet].values())) + 1
        self._vocabulary = symbols**self.order

        # The z-score and percentile need the corpus's own score distribution, so the
        # fitted sequences are re-scored against the finished model.
        self._reference_scores = tuple(sorted(self._surprisal(reduced) for reduced in observed))
        return self

    def _surprisal(self, reduced: str) -> float:
        """Mean bits of surprise per n-mer under the Laplace-smoothed background."""
        counts = nmer_counts(reduced, self.order)
        total = sum(counts.values())
        if total <= 0 or not self.is_fitted:
            return 0.0

        denominator = self._total + self._vocabulary
        accumulated = 0.0
        for nmer, count in counts.items():
            probability = (self._counts.get(nmer, 0) + 1) / denominator
            accumulated += count * -math.log2(probability)
        return accumulated / total

    def score(self, sequence: str) -> GrammarAnomaly:
        """Score one sequence's grammar against the fitted background.

        A high surprisal means the region is written with n-mer statistics the corpus
        rarely produces. The percentile is against the corpus's own scores, so it answers
        the question a reader actually has: how many ordinary regions look at least this
        unusual?
        """
        reduced = reduce_sequence(sequence, alphabet=self.alphabet)
        surprisal = self._surprisal(reduced)
        if not self._reference_scores:
            return GrammarAnomaly(surprisal_per_residue=surprisal, z_score=0.0, percentile=0.0)

        mean = sum(self._reference_scores) / len(self._reference_scores)
        variance = sum((value - mean) ** 2 for value in self._reference_scores) / len(self._reference_scores)
        deviation = variance**0.5
        z_score = (surprisal - mean) / deviation if deviation > 0 else 0.0
        at_or_below = sum(1 for value in self._reference_scores if value <= surprisal)
        return GrammarAnomaly(
            surprisal_per_residue=surprisal,
            z_score=z_score,
            percentile=at_or_below / len(self._reference_scores),
        )


def profile_regions(
    regions: Sequence[tuple[str, str]], *, alphabet: str = DEFAULT_ALPHABET
) -> dict[str, GrammarProfile]:
    """Profile a batch of (identifier, sequence) pairs."""
    return {identifier: profile_sequence(sequence, alphabet=alphabet) for identifier, sequence in regions if sequence}


# Window for the conditional-complexity profile. Wide enough that a compressed window is
# larger than the compressor's own per-call overhead, narrow enough to localise a motif.
# The measured surprisal of a single residue is dominated by that overhead, which is why
# the profile is windowed rather than computed per position.
DEFAULT_PROFILE_WINDOW = 20

# Step between window starts. Overlapping windows smooth the profile by construction,
# without assuming the noise is Gaussian - which it is not, being both autocorrelated and
# bounded below by the compressor's block structure. A filter (Savitzky-Golay, Gaussian)
# would impose exactly that assumption.
DEFAULT_PROFILE_STEP = 5

# Windows needed before a spread is meaningful.
MIN_WINDOWS_FOR_SPREAD = 2


@dataclass(frozen=True, slots=True)
class ComplexityProfile:
    """Position-resolved conditional complexity along one sequence."""

    window: int
    step: int
    starts: tuple[int, ...]
    bits_per_residue: tuple[float, ...]
    length: int

    @property
    def mean(self) -> float:
        """Mean conditional complexity, in bits per residue."""
        return sum(self.bits_per_residue) / len(self.bits_per_residue) if self.bits_per_residue else 0.0

    @property
    def peak(self) -> float:
        """Most surprising window, in bits per residue."""
        return max(self.bits_per_residue, default=0.0)

    @property
    def trough(self) -> float:
        """Least surprising window - the most predictable stretch."""
        return min(self.bits_per_residue, default=0.0)

    @property
    def spread(self) -> float:
        """Standard deviation across windows.

        A protein of uniform grammar has a flat profile; one that switches between an
        ordered domain and a repetitive linker has a wide one. This is the quantity that
        distinguishes architecture without measuring length.
        """
        if len(self.bits_per_residue) < MIN_WINDOWS_FOR_SPREAD:
            return 0.0
        mean = self.mean
        variance = sum((value - mean) ** 2 for value in self.bits_per_residue) / len(self.bits_per_residue)
        return variance**0.5

    @property
    def dynamic_range(self) -> float:
        """Difference between the most and least surprising window, in bits per residue.

        This is the "unusual for itself" quantity, expressed as a difference rather than a
        ratio. A ratio against the sequence's own mean was the obvious formulation and is
        unusable: on a highly repetitive protein the mean approaches zero and the ratio
        explodes - on a pure repeat it reached 31 against 2.7 for the same sequence
        repeated fewer times, which is an artefact of the denominator, not a property of
        the protein. Repetitive is precisely what class B's G/F-rich regions are, so the
        instability would have struck exactly where the feature mattered.

        Both terms are already per-residue, so the difference carries no absolute length.
        """
        return self.peak - self.trough

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "window": self.window,
            "step": self.step,
            "length": self.length,
            "mean_bits_per_residue": round(self.mean, 4),
            "peak_bits_per_residue": round(self.peak, 4),
            "trough_bits_per_residue": round(self.trough, 4),
            "spread": round(self.spread, 4),
            "dynamic_range": round(self.dynamic_range, 4),
        }


def conditional_complexity_profile(
    sequence: str,
    *,
    alphabet: str = DEFAULT_ALPHABET,
    window: int = DEFAULT_PROFILE_WINDOW,
    step: int = DEFAULT_PROFILE_STEP,
) -> ComplexityProfile:
    """Complexity of each window *given everything before it*, normalised per residue.

    The whole-region measures elsewhere in this module answer "how is this region built?".
    This answers "how surprising is each part, given what the protein has already said?" -
    which is the quantity that distinguishes a repetitive linker following an ordered
    domain from the same composition scattered uniformly.

    For a window starting at ``n``, the cost is the extra compressed length of the prefix
    when that window is appended:

        bits(n) = [ C(prefix[0 : n + w]) - C(prefix[0 : n]) ] / w

    The subtraction is what makes it conditional: whatever the compressor already learned
    from residues ``0 : n`` is free, so a repeat of something seen earlier costs almost
    nothing while a genuinely new stretch costs full price. That is the sense in which
    position 46 is scored "given the previous 45".

    Two normalisations keep this free of absolute length. Dividing by ``w`` gives bits per
    residue rather than bits per window, and :attr:`ComplexityProfile.relative_peak`
    expresses the extreme relative to the sequence's own mean - so a long protein cannot
    score higher merely by being long, which is exactly the artefact that made a raw
    length feature a leak in the classifier.

    Returns an empty profile when the sequence is shorter than two windows, since the
    first window is spent establishing the prefix that the rest are conditioned on.
    """
    reduced = reduce_sequence(sequence, alphabet=alphabet)
    width = max(1, window)
    stride = max(1, step)
    if len(reduced) < 2 * width:
        return ComplexityProfile(window=width, step=stride, starts=(), bits_per_residue=(), length=len(sequence))

    # Windows begin one full window in, so every one is conditioned on a real prefix.
    # A window at position zero has nothing before it, so it is charged the compressor's
    # entire stream header - measured at 12.0 bits per residue regardless of content,
    # which made it the peak of every sequence and turned that statistic into a
    # measurement of LZMA rather than of the protein.
    starts: list[int] = []
    bits: list[float] = []
    for start in range(width, len(reduced) - width + 1, stride):
        prefix = _compressed_size(reduced[:start])
        extended = _compressed_size(reduced[: start + width])
        starts.append(start)
        # Clamped at zero: a compressor can occasionally shrink on more input when a new
        # block boundary lets it re-encode, which is an artefact rather than negative
        # information.
        bits.append(max(0.0, (extended - prefix) * 8.0 / width))

    return ComplexityProfile(
        window=width,
        step=stride,
        starts=tuple(starts),
        bits_per_residue=tuple(bits),
        length=len(sequence),
    )


# Charge assignment at physiological pH. His is left neutral: its pKa is near 6, so it is
# only partly protonated at pH 7.4, and counting it as a full positive would overstate the
# charge of every histidine-rich linker.
_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")


@dataclass(frozen=True, slots=True)
class ChargeProfile:
    """Local and global charge character of one sequence.

    Both scales are kept because they answer different questions. Net charge over a whole
    protein averages a basic patch against an acidic one and can report zero for a strongly
    polarised sequence; the windowed measures find the patches. Charge *segregation* in
    particular is what distinguishes a linker that merely contains charged residues from
    one that separates them into blocks, and block separation is what drives the
    electrostatic behaviour of a disordered region.
    """

    length: int
    window: int
    net_charge: float
    fraction_charged: float
    mean_window_charge: float
    max_window_charge: float
    min_window_charge: float
    charge_segregation: float

    @property
    def net_charge_per_residue(self) -> float:
        """Net charge normalised by length, so proteins of any size compare."""
        return self.net_charge / self.length if self.length else 0.0

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the report."""
        return {
            "length": self.length,
            "window": self.window,
            "net_charge": round(self.net_charge, 4),
            "net_charge_per_residue": round(self.net_charge_per_residue, 5),
            "fraction_charged": round(self.fraction_charged, 4),
            "mean_window_charge": round(self.mean_window_charge, 4),
            "max_window_charge": round(self.max_window_charge, 4),
            "min_window_charge": round(self.min_window_charge, 4),
            "charge_segregation": round(self.charge_segregation, 4),
        }


def charge_profile(sequence: str, *, window: int = DEFAULT_PROFILE_WINDOW) -> ChargeProfile:
    """Measure charge globally and in sliding windows.

    ``charge_segregation`` is the spread of the per-window net charge divided by the
    fraction of residues that carry any charge. Dividing by that fraction separates two
    situations a raw spread confuses: a sequence with few charges scattered thinly, and a
    sequence with the same number of charges gathered into oppositely charged blocks. Only
    the second has a large segregation value, and only the second behaves like a
    polyampholyte.

    Every quantity except ``net_charge`` is per-residue or dimensionless, so none of them
    encodes absolute length.
    """
    upper = sequence.upper()
    length = len(upper)
    if not length:
        return ChargeProfile(0, window, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    charges = [1.0 if residue in _POSITIVE else -1.0 if residue in _NEGATIVE else 0.0 for residue in upper]
    net = sum(charges)
    fraction_charged = sum(1 for value in charges if value != 0.0) / length

    width = max(1, min(window, length))
    window_means: list[float] = []
    running = sum(charges[:width])
    window_means.append(running / width)
    for index in range(width, length):
        running += charges[index] - charges[index - width]
        window_means.append(running / width)

    mean_window = sum(window_means) / len(window_means)
    variance = sum((value - mean_window) ** 2 for value in window_means) / len(window_means)
    segregation = (variance**0.5) / fraction_charged if fraction_charged > 0 else 0.0

    return ChargeProfile(
        length=length,
        window=width,
        net_charge=net,
        fraction_charged=fraction_charged,
        mean_window_charge=mean_window,
        max_window_charge=max(window_means),
        min_window_charge=min(window_means),
        charge_segregation=segregation,
    )
