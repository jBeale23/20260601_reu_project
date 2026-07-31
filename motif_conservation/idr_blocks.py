"""Complexity-adaptive IDR / G/F block segmentation (Kolmogorov proxy)."""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

from motif_conservation.charge_alphabet import composition_symbol

_MIN_ENTROPY_SAMPLES = 2
_DOMINANT_SYMBOL_FRACTION = 0.6
_ENRICHED_SYMBOL_FRACTION = 0.4


@dataclass(frozen=True, slots=True)
class SequenceBlock:
    """One compositional block in an IDR-like region."""

    start: int  # 0-based
    end: int  # exclusive
    block_type: str
    sequence: str


def reduced_composition_string(sequence: str) -> str:
    """Map sequence to G/+/-/O alphabet."""
    return "".join(composition_symbol(aa) for aa in sequence)


def local_entropy(symbols: str, start: int, end: int) -> float:
    """Shannon entropy of a symbol window."""
    window = symbols[start:end]
    if not window:
        return 0.0
    counts = Counter(window)
    length = len(window)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def compression_ratio(text: str) -> float:
    """Zlib compression ratio as a Kolmogorov-complexity proxy (higher = more complex)."""
    if not text:
        return 0.0
    raw = text.encode("ascii", errors="ignore")
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)


def segment_composition_blocks(
    sequence: str,
    *,
    min_block: int = 4,
    entropy_window: int = 8,
    entropy_delta: float = 0.75,
) -> list[SequenceBlock]:
    """Segment a sequence into compositional blocks via entropy change-points.

    Adjacent residues with the same reduced symbol are merged; change-points are
    reinforced when local entropy shifts by ``entropy_delta``.
    """
    if not sequence:
        return []

    symbols = reduced_composition_string(sequence)
    boundaries = [0]
    for i in range(1, len(symbols)):
        symbol_change = symbols[i] != symbols[i - 1]
        left = max(0, i - entropy_window)
        right = min(len(symbols), i + entropy_window)
        ent_before = local_entropy(symbols, left, i) if i - left >= _MIN_ENTROPY_SAMPLES else 0.0
        ent_after = local_entropy(symbols, i, right) if right - i >= _MIN_ENTROPY_SAMPLES else 0.0
        entropy_change = abs(ent_after - ent_before) >= entropy_delta
        if (symbol_change or entropy_change) and i - boundaries[-1] >= min_block:
            boundaries.append(i)
    boundaries.append(len(symbols))

    blocks: list[SequenceBlock] = []
    for start, end in pairwise(boundaries):
        chunk = sequence[start:end]
        symbol_chunk = symbols[start:end]
        block_type = _dominant_block_type(symbol_chunk)
        blocks.append(SequenceBlock(start=start, end=end, block_type=block_type, sequence=chunk))
    return blocks


def _dominant_block_type(symbols: str) -> str:
    if not symbols:
        return "empty"
    counts = Counter(symbols)
    dominant, count = max(counts.items(), key=lambda item: item[1])
    if count / len(symbols) >= _DOMINANT_SYMBOL_FRACTION:
        mapping = {"G": "poly_gf", "+": "positive", "-": "negative", "O": "mixed", "X": "gap"}
        return mapping.get(dominant, dominant)
    if counts.get("G", 0) / len(symbols) >= _ENRICHED_SYMBOL_FRACTION:
        return "gf_rich"
    if (counts.get("+", 0) + counts.get("-", 0)) / len(symbols) >= _ENRICHED_SYMBOL_FRACTION:
        return "charged"
    return "mixed"


def block_grammar(blocks: list[SequenceBlock]) -> str:
    """Ordered block-type string, e.g. poly_gf>charged>mixed."""
    if not blocks:
        return ""
    return ">".join(block.block_type for block in blocks)


def grammar_recurrence(grammars: list[str]) -> str:
    """Most common non-empty grammar across homologs."""
    counts = Counter(grammar for grammar in grammars if grammar)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]
