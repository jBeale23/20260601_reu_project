"""Sequence-grammar signals over a run, as an independent check on the novelty call.

The novelty score elsewhere is architectural: it asks whether a protein's *domains* are
arranged unusually. This asks whether its unalignable sequence is *written* unusually.
The two are independent - a protein can carry a textbook architecture and a linker unlike
anything else in the corpus, or an odd architecture built entirely from ordinary parts.

Reporting them side by side is the point. If architectural novelty candidates were also
grammatically unusual, that would be corroboration from a measurement that shares none of
the same assumptions. If they are not, the novelty call rests on architecture alone, and
the report should say so rather than let a reader assume otherwise.
"""

from __future__ import annotations

from statistics import mean, median
from typing import TYPE_CHECKING

from domain_layout.constants import ROUTE_SHARK
from domain_layout.grammar import CorpusGrammar, profile_sequence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain_layout.pipeline import ProteinLayout

# Regions sampled to fit the background n-mer model. The corpus only has to characterise
# what an ordinary unalignable region looks like, and the estimate is stable well below
# the full set, which at proteome scale is millions of regions.
MAX_BACKGROUND_REGIONS = 20000

# Percentile above which a region's grammar counts as anomalous against the corpus.
GRAMMAR_ANOMALY_PERCENTILE = 0.99


def _shark_regions(layouts: Sequence[ProteinLayout]) -> list[tuple[str, str]]:
    """Every unalignable region, tagged with the accession it came from."""
    return [
        (layout.record.accession, region.sequence)
        for layout in layouts
        for region in layout.regions
        if region.route == ROUTE_SHARK and region.sequence
    ]


def grammar_summary(layouts: Sequence[ProteinLayout], *, seed: int = 0) -> dict[str, object]:
    """Summarise sequence grammar across a run and cross it with the novelty call."""
    regions = _shark_regions(layouts)
    if not regions:
        return {"n_regions": 0, "note": "no unalignable regions to profile"}

    sample = regions[:MAX_BACKGROUND_REGIONS]
    background = CorpusGrammar().fit([sequence for _accession, sequence in sample])

    profiles = [profile_sequence(sequence, seed=seed) for _accession, sequence in sample]
    reliable = [profile for profile in profiles if profile.is_reliable]

    anomalous: set[str] = set()
    for accession, sequence in sample:
        if background.score(sequence).percentile >= GRAMMAR_ANOMALY_PERCENTILE:
            anomalous.add(accession)

    novel = {layout.record.accession for layout in layouts if layout.classification.novel_class_candidate}
    sampled_accessions = {accession for accession, _sequence in sample}
    novel_in_sample = novel & sampled_accessions

    overlap = len(novel_in_sample & anomalous)
    # What the overlap would be if the two signals were independent, so a reader can see
    # at once whether the corroboration is real or arithmetic.
    expected = len(novel_in_sample) * len(anomalous) / len(sampled_accessions) if sampled_accessions else 0.0

    return {
        "n_regions_profiled": len(sample),
        "n_regions_total": len(regions),
        "n_reliable": len(reliable),
        "median_normalized_entropy": round(median(p.normalized_entropy for p in reliable), 4) if reliable else 0.0,
        "median_order_score": round(median(p.order_score for p in reliable), 4) if reliable else 0.0,
        "mean_compression_ratio": round(mean(p.compression_ratio for p in reliable), 4) if reliable else 0.0,
        "n_low_complexity": sum(1 for p in reliable if p.is_low_complexity),
        "n_compositionally_degenerate": sum(1 for p in reliable if p.is_compositionally_degenerate),
        "anomaly_percentile": GRAMMAR_ANOMALY_PERCENTILE,
        "n_grammar_anomalous_proteins": len(anomalous),
        "novelty_cross_check": {
            "n_novel_candidates_in_sample": len(novel_in_sample),
            "n_also_grammar_anomalous": overlap,
            "expected_if_independent": round(expected, 2),
            "enrichment": round(overlap / expected, 3) if expected > 0 else 0.0,
            "interpretation": (
                "Enrichment near 1.0 means architectural novelty and grammatical novelty "
                "are independent, so the novelty call rests on architecture alone."
            ),
        },
    }
