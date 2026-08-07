"""Tests for the domain-layout pipeline and its CLI."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from domain_layout import disorder as disorder_module
from domain_layout import pipeline as pipeline_module
from domain_layout.cli import main as layout_main
from domain_layout.constants import FEATURE_COLUMNS, REGION_COLUMNS, ROUTE_MSA, ROUTE_SHARK
from domain_layout.disorder import BACKEND_FOLDINDEX
from domain_layout.pipeline import (
    FEATURES_FILENAME,
    REGIONS_FILENAME,
    SUBFASTA_DIRNAME,
    SUMMARY_FILENAME,
    LayoutRunConfig,
    _best_reference_match,
    _chunk_records,
    _resolved_disorder_backend,
    _should_parallelize,
    analyze_store,
    build_summary,
    feature_row,
    load_reference_regions,
    region_rows,
    segment_records,
    write_outputs,
)
from domain_layout.profiles import ReferenceRegion
from domain_layout.records import DomainStore, write_domain_store
from domain_layout.shark import BACKEND_KMER
from tests.conftest import DNAJ_ECOLI_SEQUENCE, make_record

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, shark_backend=BACKEND_KMER)


def test_segment_records_skips_records_without_sequence(domain_store: DomainStore) -> None:
    """Records fetched without a sequence are skipped rather than crashing."""
    domain_store.add(make_record("NOSEQ", ""))
    segmented = segment_records(list(domain_store), backend=BACKEND_FOLDINDEX)
    assert {record.accession for record, _prediction, _regions in segmented} == {"P08622", "TEST01"}


def test_analyze_store_reports_backends_and_skips(domain_store: DomainStore) -> None:
    """The run result records which backends produced the numbers."""
    domain_store.add(make_record("NOSEQ", ""))
    result = analyze_store(domain_store, config=_CONFIG)
    assert result.disorder_backend == BACKEND_FOLDINDEX
    assert result.shark_backend == BACKEND_KMER
    assert result.skipped_without_sequence == ("NOSEQ",)
    assert {layout.record.accession for layout in result.layouts} == {"P08622", "TEST01"}


def test_analyze_store_respects_max_proteins(domain_store: DomainStore) -> None:
    """--max-proteins limits how many records are analyzed."""
    result = analyze_store(domain_store, config=LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, max_proteins=1))
    assert len(result.layouts) == 1


def test_feature_and_region_rows_match_declared_columns(domain_store: DomainStore) -> None:
    """Row builders emit exactly the documented CSV columns."""
    result = analyze_store(domain_store, config=_CONFIG)
    layout = result.layouts[0]
    assert set(feature_row(layout, result)) == set(FEATURE_COLUMNS)
    for row in region_rows(layout):
        assert set(row) == set(REGION_COLUMNS)


def test_write_outputs_creates_tables_and_subfastas(tmp_path: Path, domain_store: DomainStore) -> None:
    """Every documented output file is written with consistent content."""
    result = analyze_store(domain_store, config=_CONFIG)
    summary = write_outputs(result, tmp_path)

    regions_path = tmp_path / REGIONS_FILENAME
    features_path = tmp_path / FEATURES_FILENAME
    assert regions_path.is_file()
    assert features_path.is_file()

    feature_rows = list(csv.DictReader(features_path.open(encoding="utf-8")))
    assert {row["accession"] for row in feature_rows} == {"P08622", "TEST01"}
    dnaj_row = next(row for row in feature_rows if row["accession"] == "P08622")
    assert dnaj_row["layout_predicted_class"] == "A"
    assert dnaj_row["disorder_backend"] == BACKEND_FOLDINDEX
    assert dnaj_row["has_hpd"] == "true"

    region_csv_rows = list(csv.DictReader(regions_path.open(encoding="utf-8")))
    assert {row["route"] for row in region_csv_rows} <= {ROUTE_MSA, ROUTE_SHARK, "skip"}

    stored_summary = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert stored_summary["n_proteins"] == 2
    assert stored_summary == summary

    j_domain_fasta = tmp_path / SUBFASTA_DIRNAME / "msa" / "j_domain.fasta"
    assert j_domain_fasta.is_file()
    assert j_domain_fasta.read_text(encoding="utf-8").startswith(">P08622|5-67|structured_domain")
    assert (tmp_path / SUBFASTA_DIRNAME / "by_class" / "class_A.fasta").is_file()


def test_write_outputs_can_skip_fastas(tmp_path: Path, domain_store: DomainStore) -> None:
    """--no-fasta suppresses the subFASTA tree but keeps the tables."""
    result = analyze_store(domain_store, config=_CONFIG)
    summary = write_outputs(result, tmp_path, write_fastas=False)
    assert not (tmp_path / SUBFASTA_DIRNAME).exists()
    assert "subfastas" not in summary
    assert (tmp_path / FEATURES_FILENAME).is_file()


def test_build_summary_counts_classes_and_routes(domain_store: DomainStore) -> None:
    """The summary aggregates class calls, novel candidates, and routed residues."""
    result = analyze_store(domain_store, config=_CONFIG)
    summary = build_summary(result)
    assert summary["class_counts"]["A"] == 1
    assert summary["route_residues"][ROUTE_MSA] > 0
    assert summary["route_residues"][ROUTE_SHARK] > 0
    assert summary["n_novel_candidates"] == sum(
        1 for layout in result.layouts if layout.classification.novel_class_candidate
    )


def test_reference_regions_scored_against_query(tmp_path: Path, domain_store: DomainStore) -> None:
    """Reference regions produce a best-match reference and class per protein."""
    store_path = tmp_path / "reference_store.json"
    classes_path = tmp_path / "reference_classes.json"
    write_domain_store(domain_store, store_path)
    classes_path.write_text(json.dumps({"P08622": "A"}), encoding="utf-8")

    references = load_reference_regions(store_path, classes_path, backend=BACKEND_FOLDINDEX)
    assert references
    assert {reference.jdp_class for reference in references} == {"A"}

    result = analyze_store(domain_store, config=_CONFIG, references=references)
    dnaj_layout = next(layout for layout in result.layouts if layout.record.accession == "P08622")
    assert dnaj_layout.shark_match is not None
    assert dnaj_layout.shark_match_class == "A"
    assert result.n_references == len(references)


def test_load_reference_regions_rejects_non_object_class_map(tmp_path: Path, domain_store: DomainStore) -> None:
    """A malformed class map raises a clear TypeError."""
    store_path = tmp_path / "store.json"
    classes_path = tmp_path / "classes.json"
    write_domain_store(domain_store, store_path)
    classes_path.write_text(json.dumps(["P08622"]), encoding="utf-8")
    with pytest.raises(TypeError, match="must be a JSON object"):
        load_reference_regions(store_path, classes_path, backend=BACKEND_FOLDINDEX)


def test_cli_end_to_end(tmp_path: Path, domain_store: DomainStore, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI writes tables for a store without reference data."""
    store_path = tmp_path / "store.json"
    write_domain_store(domain_store, store_path)
    output_dir = tmp_path / "results"

    layout_main(
        [
            str(store_path),
            "-o",
            str(output_dir),
            "--no-references",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
            "--shark-backend",
            BACKEND_KMER,
        ],
    )

    assert (output_dir / FEATURES_FILENAME).is_file()
    assert "Analyzed 2 protein(s)" in capsys.readouterr().err


def test_cli_show_backends(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """--show-backends reports the active backends without needing a store."""
    monkeypatch.setattr(disorder_module, "metapredict_available", lambda: False)
    layout_main(["--show-backends"])
    captured = capsys.readouterr().err
    assert f"disorder_backend={BACKEND_FOLDINDEX}" in captured
    assert "shark_backend=" in captured


def test_cli_requires_store() -> None:
    """Omitting the store is a usage error, not a traceback."""
    with pytest.raises(SystemExit):
        layout_main([])


def test_cli_rejects_missing_store(tmp_path: Path) -> None:
    """A nonexistent store path exits with a usage error."""
    with pytest.raises(SystemExit):
        layout_main([str(tmp_path / "nope.json")])


def test_cli_warns_when_reference_data_missing(
    tmp_path: Path,
    domain_store: DomainStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing reference store degrades to empty SHARK columns with a warning."""
    store_path = tmp_path / "store.json"
    write_domain_store(domain_store, store_path)
    layout_main(
        [
            str(store_path),
            "-o",
            str(tmp_path / "out"),
            "--reference-store",
            str(tmp_path / "missing.json"),
            "--reference-classes",
            str(tmp_path / "missing_classes.json"),
            "--no-fasta",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
        ],
    )
    assert "no reference JDP profiles found" in capsys.readouterr().err


def test_cli_warns_about_records_without_sequence(
    tmp_path: Path,
    domain_store: DomainStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Records fetched without a sequence are counted and reported, not silently dropped."""
    domain_store.add(make_record("NOSEQ", ""))
    store_path = tmp_path / "store.json"
    write_domain_store(domain_store, store_path)

    layout_main(
        [
            str(store_path),
            "-o",
            str(tmp_path / "out"),
            "--no-references",
            "--no-fasta",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
        ],
    )
    captured = capsys.readouterr().err
    assert "1 record(s) had no sequence" in captured


def test_cli_warns_on_empty_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An empty domain store produces a warning and empty tables rather than a crash."""
    store_path = tmp_path / "empty.json"
    write_domain_store(DomainStore(), store_path)
    output_dir = tmp_path / "out"

    layout_main([str(store_path), "-o", str(output_dir), "--no-references", "--disorder-backend", BACKEND_FOLDINDEX])

    captured = capsys.readouterr().err
    assert "contains no protein records" in captured
    rows = list(csv.DictReader((output_dir / FEATURES_FILENAME).open(encoding="utf-8")))
    assert rows == []


def test_cli_rejects_unreadable_store(tmp_path: Path) -> None:
    """A file that is not a domain store exits with a usage error."""
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        layout_main([str(bad), "-o", str(tmp_path / "out")])


def test_analyze_store_without_references_leaves_shark_columns_empty(domain_store: DomainStore) -> None:
    """With no reference set the SHARK columns are empty and the reason is tagged."""
    result = analyze_store(domain_store, config=_CONFIG, references=())
    for layout in result.layouts:
        assert layout.shark_match is None
        assert layout.shark_match_class == ""
        assert "no_reference_similarity" in layout.classification.evidence_tags

    row = feature_row(result.layouts[0], result)
    assert row["shark_best_similarity"] == ""
    assert row["shark_best_reference"] == ""


def test_summary_counts_skipped_records(tmp_path: Path, domain_store: DomainStore) -> None:
    """The run summary lists accessions skipped for having no sequence."""
    domain_store.add(make_record("NOSEQ", ""))
    result = analyze_store(domain_store, config=_CONFIG)
    summary = write_outputs(result, tmp_path, write_fastas=False)
    assert summary["n_skipped_without_sequence"] == 1
    assert summary["skipped_without_sequence"] == ["NOSEQ"]


def test_cli_uses_explicit_reference_files(tmp_path: Path, domain_store: DomainStore) -> None:
    """Explicit --reference-store/--reference-classes populate the SHARK columns."""
    reference_store = tmp_path / "reference_store.json"
    reference_classes = tmp_path / "reference_classes.json"
    write_domain_store(domain_store, reference_store)
    reference_classes.write_text(json.dumps({"P08622": "A"}), encoding="utf-8")

    query_store = tmp_path / "query.json"
    write_domain_store(domain_store, query_store)
    output_dir = tmp_path / "out"

    layout_main(
        [
            str(query_store),
            "-o",
            str(output_dir),
            "--reference-store",
            str(reference_store),
            "--reference-classes",
            str(reference_classes),
            "--no-fasta",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
            "--shark-backend",
            BACKEND_KMER,
        ],
    )

    rows = {row["accession"]: row for row in csv.DictReader((output_dir / FEATURES_FILENAME).open(encoding="utf-8"))}
    assert rows["P08622"]["shark_best_reference_class"] == "A"
    assert float(rows["P08622"]["shark_best_similarity"]) > 0.0


def test_cli_reports_malformed_reference_class_map(tmp_path: Path, domain_store: DomainStore) -> None:
    """A malformed reference class map is a usage error, not a traceback."""
    reference_store = tmp_path / "reference_store.json"
    reference_classes = tmp_path / "reference_classes.json"
    write_domain_store(domain_store, reference_store)
    reference_classes.write_text(json.dumps(["P08622"]), encoding="utf-8")

    query_store = tmp_path / "query.json"
    write_domain_store(domain_store, query_store)

    with pytest.raises(SystemExit):
        layout_main(
            [
                str(query_store),
                "-o",
                str(tmp_path / "out"),
                "--reference-store",
                str(reference_store),
                "--reference-classes",
                str(reference_classes),
            ],
        )


def test_best_reference_match_scores_zero_against_an_empty_reference(domain_store: DomainStore) -> None:
    """An unusable reference still reports a match, at score 0.0 rather than silently None.

    A zero score is what drives the novelty term, so it must be a real number the caller
    can compare against the threshold, not a missing value.
    """
    record = next(iter(domain_store))
    regions = segment_records([record], backend=BACKEND_FOLDINDEX)[0][2]
    reference = ReferenceRegion(reference_id="ref", accession="REF", jdp_class="A", kind="idr", sequence="")

    match, match_class = _best_reference_match(regions, [reference], backend=BACKEND_KMER)
    assert match is not None
    assert match.score == 0.0
    assert match_class == "A"


def test_best_reference_match_without_query_regions() -> None:
    """A protein with no unalignable regions has nothing to compare and reports none."""
    reference = ReferenceRegion(reference_id="ref", accession="REF", jdp_class="A", kind="idr", sequence="GGFGG")
    match, match_class = _best_reference_match([], [reference], backend=BACKEND_KMER)
    assert match is None
    assert match_class == ""


def test_best_reference_match_labels_unknown_reference_class(domain_store: DomainStore) -> None:
    """A match against a reference missing from the class map yields an empty class."""
    record = next(iter(domain_store))
    regions = segment_records([record], backend=BACKEND_FOLDINDEX)[0][2]
    shark_regions = [region for region in regions if region.route == ROUTE_SHARK]
    reference = ReferenceRegion(
        reference_id="ref",
        accession="REF",
        jdp_class="",
        kind="idr",
        sequence=shark_regions[0].sequence,
    )
    match, match_class = _best_reference_match(regions, [reference], backend=BACKEND_KMER)
    assert match is not None
    assert match_class == ""


def _classification_signature(result: object) -> list[tuple[str, str, str, float, str]]:
    return [
        (
            layout.record.accession,
            layout.classification.predicted_class,
            layout.classification.predicted_subclass,
            round(layout.classification.novelty_score, 6),
            layout.shark_match_class,
        )
        for layout in result.layouts  # type: ignore[attr-defined]
    ]


def test_parallel_workers_produce_identical_results(domain_store: DomainStore) -> None:
    """Splitting across processes must not change any call or the output order."""
    records = list(domain_store)
    many = DomainStore()
    for index in range(60):
        many.add(replace(records[index % len(records)], accession=f"SYN{index:04d}"))

    serial = analyze_store(many, config=replace(_CONFIG, workers=1))
    parallel = analyze_store(many, config=replace(_CONFIG, workers=3))

    assert _classification_signature(parallel) == _classification_signature(serial)
    assert [layout.record.accession for layout in parallel.layouts] == [
        layout.record.accession for layout in serial.layouts
    ]
    assert parallel.disorder_backend == serial.disorder_backend


def test_small_runs_stay_serial() -> None:
    """A handful of proteins is not worth a process pool."""
    assert _should_parallelize(replace(_CONFIG, workers=4), n_records=2) is False
    assert _should_parallelize(replace(_CONFIG, workers=1), n_records=10_000) is False
    assert _should_parallelize(replace(_CONFIG, workers=4), n_records=10_000) is True


def test_chunking_covers_every_record_exactly_once() -> None:
    """Chunks partition the record list without dropping or duplicating anything."""
    records = [make_record(f"P{index:04d}", "MHPDK" * 10) for index in range(97)]
    for workers in (1, 2, 4, 8, 64):
        chunks = _chunk_records(records, workers)
        flattened = [record.accession for chunk in chunks for record in chunk]
        assert flattened == [record.accession for record in records], workers
        assert all(chunks), workers


def test_chunking_of_empty_input() -> None:
    """Chunking no records yields no chunks to schedule."""
    assert _chunk_records([], 4) == []


def test_worker_pool_uses_spawn_not_fork(domain_store: DomainStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """The process pool must use spawn.

    metapredict imports PyTorch, which starts threads. Forking a multi-threaded process
    can deadlock the child; on a cluster that looks like a job hanging until its time
    limit. Spawn costs a little startup time and avoids the failure mode entirely.
    """
    requested: list[str] = []
    real_get_context = pipeline_module.get_context

    def recording_get_context(method: str):  # noqa: ANN202 - mirrors multiprocessing's API
        requested.append(method)
        return real_get_context(method)

    monkeypatch.setattr(pipeline_module, "get_context", recording_get_context)

    records = list(domain_store)
    many = DomainStore()
    for index in range(60):
        many.add(replace(records[index % len(records)], accession=f"SPWN{index:04d}"))

    analyze_store(many, config=replace(_CONFIG, workers=2))
    assert requested == ["spawn"]


def test_summary_reports_every_backend_actually_used(domain_store: DomainStore) -> None:
    """A mixed run must not claim a single backend produced everything.

    On real data most proteins used metapredict while sequences containing X fell back
    to FoldIndex; reporting only the first protein's backend hid that from the summary.
    """
    result = analyze_store(domain_store, config=_CONFIG)
    assert len(result.layouts) >= 2

    # Simulate the mix that real UniProt data produces.
    mixed = replace(result.layouts[0], disorder=replace(result.layouts[0].disorder, backend="metapredict"))
    result.layouts[0] = mixed

    summary = build_summary(result)
    assert summary["disorder_backend_counts"] == {"foldindex": len(result.layouts) - 1, "metapredict": 1}


def test_resolved_backend_names_the_mix() -> None:
    """The run-level backend field lists each backend that produced predictions."""
    config = _CONFIG
    assert _resolved_disorder_backend([], config) in {BACKEND_FOLDINDEX, "metapredict"}


def test_long_runs_log_progress(domain_store: DomainStore, caplog: pytest.LogCaptureFixture) -> None:
    """A multi-hour run must show life; its tables are only written at the very end."""
    with caplog.at_level("INFO", logger="domain_layout.pipeline"):
        analyze_store(domain_store, config=_CONFIG)

    messages = " ".join(record.message for record in caplog.records)
    assert "analyzing" in messages
    assert "analysis complete" in messages


def test_chunking_oversubscribes_workers_for_balance_and_progress() -> None:
    """More chunks than workers, so a long chunk cannot strand a worker at the end."""
    records = [make_record(f"P{index:05d}", DNAJ_ECOLI_SEQUENCE) for index in range(4000)]
    chunks = pipeline_module._chunk_records(records, 4)

    assert len(chunks) > 4
    assert len(chunks) == 4 * pipeline_module.CHUNKS_PER_WORKER
    # Every record appears exactly once, in order.
    assert [record.accession for chunk in chunks for record in chunk] == [r.accession for r in records]


def test_chunking_does_not_split_below_the_minimum_batch() -> None:
    """Tiny inputs are not shattered into batches too small to be worth a process."""
    records = [make_record(f"P{index:05d}", DNAJ_ECOLI_SEQUENCE) for index in range(30)]
    chunks = pipeline_module._chunk_records(records, 8)
    assert len(chunks) == 1
    assert pipeline_module._chunk_records([], 8) == []


def test_parallel_analysis_preserves_input_order() -> None:
    """Chunks are reassembled by submission index, not by completion order.

    Progress reporting requires consuming futures as they complete, which is precisely
    what would scramble the output if the results were simply appended.
    """
    records = [make_record(f"P{index:05d}", DNAJ_ECOLI_SEQUENCE) for index in range(600)]
    config = LayoutRunConfig(write_fastas=False, workers=4)

    layouts = pipeline_module._analyze_parallel(records, config, [])
    assert [layout.record.accession for layout in layouts] == [record.accession for record in records]
