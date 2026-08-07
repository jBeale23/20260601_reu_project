"""End-to-end domain-layout pipeline: store -> regions -> dual layer -> classification.

For every protein in a domain store this module:

1. predicts disorder (metapredict, else FoldIndex);
2. splits the sequence into routed regions (:mod:`domain_layout.regions`);
3. scores the SHARK-routed regions against curated reference JDP regions;
4. assigns class A/B/C, a subclass, and a novel-category candidacy score;
5. writes region/feature CSVs, a run summary, and the per-class / per-route subFASTAs.
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from math import ceil
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any

from domain_layout import disorder as disorder_backend
from domain_layout import msa as msa_backend
from domain_layout import shark as shark_backend
from domain_layout.constants import (
    FEATURE_COLUMNS,
    REGION_COLUMNS,
    ROUTE_MSA,
    ROUTE_SHARK,
)
from domain_layout.fasta import align_msa_subfastas, write_subfastas
from domain_layout.profiles import (
    LayoutClassification,
    LayoutEvidence,
    ReferenceRegion,
    build_evidence,
    classify_layout,
)
from domain_layout.records import DomainStore, ProteinDomainRecord, load_domain_store
from domain_layout.regions import Region, segment_protein

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from domain_layout.disorder import DisorderPrediction
    from domain_layout.shark import SharkMatch

REGIONS_FILENAME = "domain_layout_regions.csv"
FEATURES_FILENAME = "domain_layout_features.csv"
SUMMARY_FILENAME = "domain_layout_summary.json"
SUBFASTA_DIRNAME = "subfastas"

# Below this many records per worker the process-pool overhead outweighs the gain.
MIN_RECORDS_PER_WORKER = 25

# How often to log progress. A full-scale run takes hours and writes its tables only at
# the end, so without this there is no way to tell a slow run from a stuck one.
PROGRESS_LOG_INTERVAL = 5000
# Chunks submitted per worker. Above one, a slow chunk cannot strand a worker at the end
# of the run, and progress is reported roughly this many times per worker.
CHUNKS_PER_WORKER = 8

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LayoutRunConfig:
    """Backend selection and output options for one layout run."""

    disorder_backend: str = disorder_backend.BACKEND_AUTO
    shark_backend: str = shark_backend.BACKEND_AUTO
    write_fastas: bool = True
    # Align the MSA-routed subFASTAs in place rather than only writing them out. Off by
    # default: on a proteome-scale run this is a second substantial compute stage, and it
    # is only meaningful once the routing itself is trusted.
    align_msa: bool = False
    msa_backend: str = msa_backend.BACKEND_AUTO
    msa_threads: int = 1
    max_aligned_per_family: int = 2000
    seed: int = 0
    max_proteins: int | None = None
    workers: int = 1


DEFAULT_RUN_CONFIG = LayoutRunConfig()


@dataclass(frozen=True, slots=True)
class ProteinLayout:
    """Everything the pipeline derived for one protein."""

    record: ProteinDomainRecord
    disorder: DisorderPrediction
    regions: tuple[Region, ...]
    evidence: LayoutEvidence
    classification: LayoutClassification
    shark_match: SharkMatch | None
    shark_match_class: str = ""


@dataclass(slots=True)
class LayoutResult:
    """All per-protein layouts plus run-level metadata."""

    layouts: list[ProteinLayout] = field(default_factory=list)
    disorder_backend: str = disorder_backend.BACKEND_FOLDINDEX
    shark_backend: str = shark_backend.BACKEND_KMER
    n_references: int = 0
    skipped_without_sequence: tuple[str, ...] = ()


def segment_records(
    records: Sequence[ProteinDomainRecord],
    *,
    backend: str = disorder_backend.BACKEND_AUTO,
) -> list[tuple[ProteinDomainRecord, DisorderPrediction, list[Region]]]:
    """Predict disorder and segment a batch of records into routed regions."""
    sequences = {record.accession: record.sequence for record in records if record.has_sequence}
    predictions = disorder_backend.predict_disorder_many(sequences, backend=backend)

    segmented: list[tuple[ProteinDomainRecord, DisorderPrediction, list[Region]]] = []
    for record in records:
        if not record.has_sequence:
            continue
        prediction = predictions[record.accession]
        segmented.append((record, prediction, segment_protein(record, prediction)))
    return segmented


def load_reference_regions(
    store_path: Path,
    classes_path: Path,
    *,
    backend: str = disorder_backend.BACKEND_AUTO,
) -> list[ReferenceRegion]:
    """Segment the curated reference JDPs and return their unalignable regions.

    Raises:
        TypeError: If the reference class map is not a JSON object.
    """
    store = load_domain_store(store_path)
    payload = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Reference class map must be a JSON object: {classes_path}"
        raise TypeError(msg)
    class_by_accession = {str(accession): str(value) for accession, value in payload.items()}

    references: list[ReferenceRegion] = []
    for record, _prediction, regions in segment_records(list(store), backend=backend):
        jdp_class = class_by_accession.get(record.accession, "")
        if not jdp_class:
            continue
        references.extend(
            ReferenceRegion(
                reference_id=region.fasta_id(),
                accession=record.accession,
                jdp_class=jdp_class,
                kind=region.kind,
                sequence=region.sequence,
            )
            for region in regions
            if region.route == ROUTE_SHARK and region.sequence
        )
    return references


def _best_reference_match(
    regions: Sequence[Region],
    references: Sequence[ReferenceRegion],
    *,
    backend: str,
) -> tuple[SharkMatch | None, str]:
    """Score a protein's unalignable regions against the reference set."""
    query_regions = [region for region in regions if region.route == ROUTE_SHARK and region.sequence]
    if not query_regions or not references:
        return None, ""

    reference_sequences = {reference.reference_id: reference.sequence for reference in references}
    class_by_reference = {reference.reference_id: reference.jdp_class for reference in references}

    best: SharkMatch | None = None
    for region in query_regions:
        match = shark_backend.best_match(region.sequence, reference_sequences, backend=backend)
        if match is not None and (best is None or match.score > best.score):
            best = match
    if best is None:
        return None, ""
    return best, class_by_reference.get(best.reference_id, "")


def analyze_store(
    store: DomainStore,
    *,
    config: LayoutRunConfig = DEFAULT_RUN_CONFIG,
    references: Sequence[ReferenceRegion] = (),
) -> LayoutResult:
    """Run the full layout analysis over every protein in a domain store.

    With ``config.workers > 1`` the proteins are split across processes. Each worker
    re-runs disorder prediction and scoring on its own slice, so results are identical
    to a serial run — only the wall time changes.
    """
    records = list(store)
    if config.max_proteins is not None:
        records = records[: config.max_proteins]

    skipped = tuple(record.accession for record in records if not record.has_sequence)
    usable = [record for record in records if record.has_sequence]

    logger.info(
        "analyzing %s protein(s) with %s worker(s); %s skipped for having no sequence",
        len(usable),
        config.workers,
        len(skipped),
    )
    if _should_parallelize(config, len(usable)):
        layouts = _analyze_parallel(usable, config, references)
    else:
        layouts = analyze_records(usable, config, references)
    logger.info("analysis complete: %s layout(s); writing tables", len(layouts))

    resolved_shark = shark_backend.active_backend(backend=config.shark_backend)
    resolved_disorder = _resolved_disorder_backend(layouts, config)

    return LayoutResult(
        layouts=layouts,
        disorder_backend=resolved_disorder,
        shark_backend=resolved_shark,
        n_references=len(references),
        skipped_without_sequence=skipped,
    )


def _resolved_disorder_backend(layouts: Sequence[ProteinLayout], config: LayoutRunConfig) -> str:
    """Name every disorder backend that actually produced a prediction.

    A run can legitimately mix backends: metapredict handles most proteins while
    sequences containing residues it cannot encode fall back to FoldIndex. Reporting
    only the first protein's backend would claim the whole run used metapredict when a
    substantial share did not.
    """
    if not layouts:
        return disorder_backend.active_backend(backend=config.disorder_backend)
    used = sorted({layout.disorder.backend for layout in layouts})
    return "+".join(used)


def analyze_records(
    records: Sequence[ProteinDomainRecord],
    config: LayoutRunConfig,
    references: Sequence[ReferenceRegion],
) -> list[ProteinLayout]:
    """Analyze one batch of records serially (also the per-worker entry point)."""
    layouts: list[ProteinLayout] = []
    for record, prediction, regions in segment_records(records, backend=config.disorder_backend):
        if layouts and len(layouts) % PROGRESS_LOG_INTERVAL == 0:
            logger.info("analyzed %s of %s protein(s) in this batch", len(layouts), len(records))
        match, match_class = _best_reference_match(regions, references, backend=config.shark_backend)
        evidence = build_evidence(record, regions)
        layouts.append(
            ProteinLayout(
                record=record,
                disorder=prediction,
                regions=tuple(regions),
                evidence=evidence,
                classification=classify_layout(evidence, shark_match=match),
                shark_match=match,
                shark_match_class=match_class,
            ),
        )
    return layouts


def _should_parallelize(config: LayoutRunConfig, n_records: int) -> bool:
    """Whether a run is big enough to be worth the process-pool overhead."""
    return config.workers > 1 and n_records >= MIN_RECORDS_PER_WORKER * 2


def _chunk_records(
    records: Sequence[ProteinDomainRecord],
    workers: int,
) -> list[list[ProteinDomainRecord]]:
    """Split records into contiguous chunks sized for a pool of ``workers``.

    More chunks than workers, for two reasons. Cost per protein scales with sequence
    length, so one chunk per worker leaves whoever drew the long proteins running alone
    at the end. And a chunk is the unit of progress: with one chunk each, a full-proteome
    run reports nothing at all until every worker is finished, which on a cluster is
    indistinguishable from a hang.
    """
    if not records:
        return []
    target = max(1, workers) * CHUNKS_PER_WORKER
    chunk_count = max(1, min(target, len(records) // MIN_RECORDS_PER_WORKER or 1))
    size = ceil(len(records) / chunk_count)
    return [list(records[start : start + size]) for start in range(0, len(records), size)]


def _analyze_parallel(
    records: Sequence[ProteinDomainRecord],
    config: LayoutRunConfig,
    references: Sequence[ReferenceRegion],
) -> list[ProteinLayout]:
    """Analyze records across a process pool, preserving input order."""
    chunks = _chunk_records(records, config.workers)
    # Each worker re-derives everything for its chunk, so a serial config is passed on
    # to stop workers from spawning pools of their own.
    worker_config = replace(config, workers=1)
    reference_list = list(references)

    # "spawn", not the Linux default "fork": metapredict pulls in PyTorch, which starts
    # threads at import. Forking a multi-threaded process can deadlock the child, which
    # on a cluster shows up as a job that hangs until it hits its time limit.
    context = get_context("spawn")

    workers = max(1, min(config.workers, len(chunks)))
    results: list[list[ProteinLayout]] = [[] for _ in chunks]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(analyze_records, chunk, worker_config, reference_list): position
            for position, chunk in enumerate(chunks)
        }
        # Report as chunks land, but reassemble by submission index so the output order
        # matches the input regardless of which worker finished first.
        for future in as_completed(futures):
            position = futures[future]
            results[position] = future.result()
            completed += 1
            logger.info(
                "analysed chunk %s of %s (%s protein(s) done)",
                completed,
                len(chunks),
                sum(len(part) for part in results),
            )
    return [layout for part in results for layout in part]


def region_rows(layout: ProteinLayout) -> list[dict[str, Any]]:
    """Render one protein's regions as CSV-ready rows."""
    return [
        {
            "accession": region.accession,
            "region_index": region.index,
            "start": region.start,
            "end": region.end,
            "length": region.length,
            "region_kind": region.kind,
            "domain_family": region.family,
            "source_signature": region.signature,
            "route": region.route,
            "mean_disorder": f"{region.mean_disorder:.4f}",
            "disorder_fraction": f"{region.disorder_fraction:.4f}",
            "net_charge": region.net_charge,
            "sequence": region.sequence,
        }
        for region in layout.regions
    ]


def feature_row(layout: ProteinLayout, result: LayoutResult) -> dict[str, Any]:
    """Render one protein's layout features as a CSV-ready row."""
    evidence = layout.evidence
    classification = layout.classification
    msa_regions = [region for region in layout.regions if region.route == ROUTE_MSA]
    shark_regions = [region for region in layout.regions if region.route == ROUTE_SHARK]

    return {
        "accession": layout.record.accession,
        "protein_name": layout.record.name,
        "protein_length": evidence.protein_length,
        "organism_name": layout.record.organism_name,
        "n_entries": len(layout.record.entries),
        "n_structured_domains": evidence.n_structured_domains,
        "domain_family_layout": evidence.domain_family_layout,
        "j_domain_position": evidence.j_domain_position,
        "has_j_domain": _boolean(evidence.has_j_domain),
        "has_hpd": _boolean(evidence.has_hpd),
        "has_dnaj_c": _boolean(evidence.has_dnaj_c),
        "has_zinc_finger_like": _boolean(evidence.has_zinc_finger_like),
        "has_gf_rich_region": _boolean(evidence.has_gf_rich_region),
        "has_transmembrane": _boolean(evidence.has_transmembrane),
        "has_signal_peptide": _boolean(evidence.has_signal_peptide),
        "idr_fraction": f"{evidence.idr_fraction:.4f}",
        "mean_disorder": f"{layout.disorder.mean_disorder:.4f}",
        "disorder_backend": layout.disorder.backend,
        "n_msa_regions": len(msa_regions),
        "n_shark_regions": len(shark_regions),
        "msa_residues": sum(region.length for region in msa_regions),
        "shark_residues": sum(region.length for region in shark_regions),
        "shark_backend": result.shark_backend,
        "shark_best_reference": layout.shark_match.reference_id if layout.shark_match else "",
        "shark_best_reference_class": layout.shark_match_class,
        "shark_best_similarity": f"{layout.shark_match.score:.4f}" if layout.shark_match else "",
        "layout_predicted_class": classification.predicted_class,
        "layout_predicted_subclass": classification.predicted_subclass,
        "layout_class_confidence": classification.class_confidence,
        "layout_novelty_score": f"{classification.novelty_score:.4f}",
        "novel_class_candidate": _boolean(classification.novel_class_candidate),
        "layout_evidence_tags": ";".join(classification.evidence_tags),
    }


def _boolean(value: bool) -> str:  # noqa: FBT001 - CSV serialization helper
    return "true" if value else "false"


def build_summary(result: LayoutResult) -> dict[str, Any]:
    """Aggregate run-level counts for the summary JSON."""
    class_counts: dict[str, int] = {}
    subclass_counts: dict[str, int] = {}
    route_residues: dict[str, int] = {ROUTE_MSA: 0, ROUTE_SHARK: 0}
    for layout in result.layouts:
        class_counts[layout.classification.predicted_class] = (
            class_counts.get(layout.classification.predicted_class, 0) + 1
        )
        subclass_counts[layout.classification.predicted_subclass] = (
            subclass_counts.get(layout.classification.predicted_subclass, 0) + 1
        )
        for region in layout.regions:
            if region.route in route_residues:
                route_residues[region.route] += region.length

    disorder_backend_counts: dict[str, int] = {}
    for layout in result.layouts:
        name = layout.disorder.backend
        disorder_backend_counts[name] = disorder_backend_counts.get(name, 0) + 1

    return {
        "n_proteins": len(result.layouts),
        "n_skipped_without_sequence": len(result.skipped_without_sequence),
        "skipped_without_sequence": list(result.skipped_without_sequence),
        "disorder_backend": result.disorder_backend,
        "disorder_backend_counts": dict(sorted(disorder_backend_counts.items())),
        "shark_backend": result.shark_backend,
        "n_reference_regions": result.n_references,
        "class_counts": dict(sorted(class_counts.items())),
        "subclass_counts": dict(sorted(subclass_counts.items())),
        "route_residues": route_residues,
        "n_novel_candidates": sum(1 for layout in result.layouts if layout.classification.novel_class_candidate),
    }


def write_outputs(
    result: LayoutResult,
    output_dir: Path,
    *,
    write_fastas: bool = True,
    config: LayoutRunConfig | None = None,
) -> dict[str, Any]:
    """Write region CSV, feature CSV, summary JSON, and subFASTAs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / REGIONS_FILENAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGION_COLUMNS)
        writer.writeheader()
        for layout in result.layouts:
            writer.writerows(region_rows(layout))

    with (output_dir / FEATURES_FILENAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for layout in result.layouts:
            writer.writerow(feature_row(layout, result))

    run_config = config or DEFAULT_RUN_CONFIG
    summary = build_summary(result)
    if write_fastas:
        subfasta_dir = output_dir / SUBFASTA_DIRNAME
        summary["subfastas"] = write_subfastas(
            {layout.record.accession: layout.regions for layout in result.layouts},
            {layout.record.accession: layout.record.sequence for layout in result.layouts},
            {layout.record.accession: layout.classification.predicted_class for layout in result.layouts},
            subfasta_dir,
        )
        if run_config.align_msa:
            logger.info("aligning MSA-routed subFASTAs")
            summary["msa_alignments"] = align_msa_subfastas(
                subfasta_dir,
                backend=run_config.msa_backend,
                threads=run_config.msa_threads,
                max_sequences=run_config.max_aligned_per_family,
                seed=run_config.seed,
            )

    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
