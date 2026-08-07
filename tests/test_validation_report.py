"""End-to-end tests for the assembled validation report and its CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from domain_layout.disorder import BACKEND_FOLDINDEX
from domain_layout.pipeline import LayoutRunConfig
from domain_layout.records import DomainStore, write_domain_store
from domain_layout.shark import BACKEND_KMER
from tests.conftest import DNAJ_ECOLI_SEQUENCE, make_entry, make_record
from validation import cli as validation_cli
from validation.cli import main as validate_main
from validation.report import PermutationSettings, build_validation_report, format_summary

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, shark_backend=BACKEND_KMER, write_fastas=False)
_FAST = PermutationSettings(novelty=20, recurrence=50, seed=0)


def _labelled_store() -> DomainStore:
    """A store with curated labels, a fragment, and several organisms."""
    store = DomainStore(source="tests")
    # Class A: J-domain + zinc finger + C-terminal domain.
    for index, organism in enumerate(["Homo sapiens", "Mus musculus", "Danio rerio"]):
        store.add(
            make_record(
                f"A{index:05d}",
                DNAJ_ECOLI_SEQUENCE,
                name="DnaJ homolog subfamily A member 1",
                source_database="reviewed",
                organism_name=organism,
                entries=(
                    make_entry("PF00226", 5, 67, integrated="IPR001623"),
                    make_entry("PF01556", 117, 143, integrated="IPR002939"),
                    make_entry("PF00684", 144, 204, integrated="IPR001305"),
                ),
            ),
        )
    # Class B: J-domain + C-terminal domain, no zinc finger.
    for index, organism in enumerate(["Homo sapiens", "Oryza sativa"]):
        store.add(
            make_record(
                f"B{index:05d}",
                DNAJ_ECOLI_SEQUENCE,
                name="DnaJ homolog subfamily B member 1",
                source_database="reviewed",
                organism_name=organism,
                entries=(
                    make_entry("PF00226", 5, 67, integrated="IPR001623"),
                    make_entry("PF01556", 117, 143, integrated="IPR002939"),
                ),
            ),
        )
    # Class C: J-domain only. The tail deliberately carries no G/F-rich block, which is
    # a class B signature the classifier detects compositionally rather than from
    # annotation alone - reusing a class A sequence here would label the fixture wrongly.
    c_sequence = DNAJ_ECOLI_SEQUENCE[:70] + ("KEQTSPEKQTSAEKQNSPEKQTSAEKQNSP" * 5)
    for index, organism in enumerate(["Homo sapiens", "Arabidopsis thaliana"]):
        store.add(
            make_record(
                f"C{index:05d}",
                c_sequence,
                name="DnaJ homolog subfamily C member 5",
                source_database="reviewed",
                organism_name=organism,
                entries=(make_entry("PF00226", 5, 67, integrated="IPR001623"),),
            ),
        )
    # An unlabelled fragment that quality control must remove.
    store.add(
        make_record(
            "FRAG01",
            DNAJ_ECOLI_SEQUENCE[:60],
            name="DnaJ domain-containing protein",
            source_database="unreviewed",
            organism_name="Uncultured bacterium",
            is_fragment=True,
            entries=(make_entry("PF00226", 5, 60, integrated="IPR001623"),),
        ),
    )
    return store


def test_report_has_every_section() -> None:
    """The report covers dataset, QC, calibration, novelty null, and recurrence."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    assert set(report) == {
        "dataset",
        "permutation_settings",
        "quality_control",
        "label_calibration",
        "novelty",
        "recurrence",
    }


def test_quality_control_removes_the_fragment() -> None:
    """The flagged fragment is excluded and the reason recorded."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    quality = report["quality_control"]
    assert quality["n_excluded"] >= 1
    assert "uniprot_fragment" in quality["exclusion_reasons"]
    assert report["dataset"]["n_after_quality_filter"] < report["dataset"]["n_analyzed"]


def test_calibration_uses_only_curated_labels_and_reports_baselines() -> None:
    """Curated proteins are scored, and trivial baselines are reported alongside."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    calibration = report["label_calibration"]

    assert calibration["n_labelled"] == 7  # the unreviewed fragment is not labelled
    names = {evaluation["name"] for evaluation in calibration["evaluations"]}
    assert names == {"domain_layout", "majority_class", "architecture_only"}

    for evaluation in calibration["evaluations"]:
        low, high = evaluation["accuracy_95ci"]
        assert 0.0 <= low <= evaluation["accuracy"] <= high <= 1.0


def test_classifier_recovers_curated_classes_on_clean_examples() -> None:
    """On textbook architectures the classifier must match curated nomenclature."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    layout = next(
        evaluation for evaluation in report["label_calibration"]["evaluations"] if evaluation["name"] == "domain_layout"
    )
    majority = next(
        evaluation
        for evaluation in report["label_calibration"]["evaluations"]
        if evaluation["name"] == "majority_class"
    )
    assert layout["accuracy"] == pytest.approx(1.0)
    assert layout["accuracy"] > majority["accuracy"]


def test_novelty_section_reports_the_null_and_threshold_sensitivity() -> None:
    """Candidate counts are reported before and after QC, with the null beside them."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    novelty = report["novelty"]

    assert novelty["candidates_after_quality_filter"] <= novelty["candidates_before_quality_filter"]
    null = novelty["permutation_null"]
    assert null["n_permutations"] == _FAST.novelty
    assert 0.0 < null["p_value"] <= 1.0
    thresholds = [row["threshold"] for row in novelty["threshold_sensitivity"]]
    assert thresholds == sorted(thresholds)


def test_recurrence_section_is_multiple_testing_corrected() -> None:
    """Every reported architecture carries an FDR-adjusted p-value."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    recurrence = report["recurrence"]
    assert recurrence["significance_fdr"] == 0.05
    for entry in recurrence["architectures"]:
        assert "p_adjusted" in entry
        assert entry["p_adjusted"] >= entry["p_value"] - 1e-9


def test_report_is_reproducible_from_the_seed() -> None:
    """Two runs with the same seed produce identical statistics."""
    first = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    second = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    assert first["novelty"]["permutation_null"] == second["novelty"]["permutation_null"]
    assert first["recurrence"]["architectures"] == second["recurrence"]["architectures"]


def test_summary_mentions_the_headline_numbers() -> None:
    """The digest states accuracy with an interval and the null comparison."""
    report = build_validation_report(_labelled_store(), config=_CONFIG, permutations=_FAST)
    summary = format_summary(report)
    assert "95% CI" in summary
    assert "permutation null" in summary
    assert "quality control" in summary


def test_cli_writes_a_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI produces a JSON report and a readable digest."""
    store_path = tmp_path / "store.json"
    write_domain_store(_labelled_store(), store_path)
    output = tmp_path / "validation_report.json"

    validate_main(
        [
            str(store_path),
            "-o",
            str(output),
            "--permutations",
            "20",
            "--recurrence-permutations",
            "50",
            "--no-references",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
            "--shark-backend",
            BACKEND_KMER,
        ],
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"]["n_analyzed"] > 0
    assert "accuracy" in report["label_calibration"]["evaluations"][0]
    assert "95% CI" in capsys.readouterr().err


def test_report_records_the_permutation_settings() -> None:
    """A report whose seed is not written down is not reproducible."""
    report = build_validation_report(
        _labelled_store(),
        config=LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, shark_backend=BACKEND_KMER, write_fastas=False),
        permutations=PermutationSettings(novelty=5, recurrence=7, seed=99),
    )
    settings = report["permutation_settings"]
    assert settings == {"seed": 99, "novelty_permutations": 5, "recurrence_permutations": 7}


def test_calibration_uses_the_quality_filtered_set_and_shows_the_unfiltered_one() -> None:
    """Scoring on filtered data is only fair if the unfiltered number is shown too."""
    store = _labelled_store()
    # A truncated fragment carrying a curated class-A name: complete enough to be
    # labelled, too incomplete to be scored fairly.
    store.add(
        replace(
            make_record(
                "FRAG01",
                DNAJ_ECOLI_SEQUENCE[:40],
                name="DnaJ homolog subfamily A member 1",
                source_database="reviewed",
                organism_name="Homo sapiens",
                entries=(make_entry("PF00226", 5, 35, integrated="IPR001623"),),
            ),
            is_fragment=True,
        ),
    )

    report = build_validation_report(
        store,
        config=LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, shark_backend=BACKEND_KMER, write_fastas=False),
        permutations=PermutationSettings(novelty=5, recurrence=5, seed=0),
    )
    calibration = report["label_calibration"]

    assert calibration["n_labelled"] == 8
    assert calibration["n_labelled_after_quality_filter"] == 7
    assert calibration["n_evaluated"] == 7
    # The unfiltered figure is reported alongside, over the full labelled set.
    assert calibration["unfiltered_comparison"]["n_evaluated"] == 8
    assert calibration["unfiltered_comparison"]["name"] == "domain_layout_unfiltered"


def test_quality_filter_does_not_silently_shrink_the_label_set_to_nothing() -> None:
    """If the filter removed every labelled protein the report must still be coherent."""
    store = DomainStore()
    store.add(
        replace(
            make_record(
                "FRAG02",
                DNAJ_ECOLI_SEQUENCE[:40],
                name="DnaJ homolog subfamily B member 1",
                source_database="reviewed",
                organism_name="Homo sapiens",
                entries=(make_entry("PF00226", 5, 35, integrated="IPR001623"),),
            ),
            is_fragment=True,
        ),
    )
    report = build_validation_report(
        store,
        config=LayoutRunConfig(disorder_backend=BACKEND_FOLDINDEX, shark_backend=BACKEND_KMER, write_fastas=False),
        permutations=PermutationSettings(novelty=3, recurrence=3, seed=0),
    )
    assert report["label_calibration"]["n_labelled_after_quality_filter"] == 0
    assert report["label_calibration"]["n_evaluated"] == 0
    for evaluation in report["label_calibration"]["evaluations"]:
        assert evaluation["accuracy"] == 0.0
    assert format_summary(report)


def test_cli_rejects_a_missing_store(tmp_path: Path) -> None:
    """A typo'd path must fail loudly, not produce an empty report."""
    with pytest.raises(SystemExit) as excinfo:
        validate_main([str(tmp_path / "absent.json"), "-o", str(tmp_path / "out.json")])
    assert excinfo.value.code == 2


def test_cli_rejects_a_malformed_store(tmp_path: Path) -> None:
    """A store that is not a domain store is a user error, not a traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a domain store"}', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        validate_main([str(bad), "-o", str(tmp_path / "out.json")])
    assert excinfo.value.code == 2


def test_cli_uses_the_bundled_references_when_available(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With references present the SHARK novelty term can fire; the run says so."""
    path = tmp_path / "store.json"
    write_domain_store(_labelled_store(), path)
    output = tmp_path / "report.json"

    validate_main(
        [
            str(path),
            "-o",
            str(output),
            "--permutations",
            "3",
            "--recurrence-permutations",
            "3",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
            "--shark-backend",
            BACKEND_KMER,
        ],
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"]["n_reference_regions"] > 0
    assert "WARNING: bundled reference JDPs not found" not in capsys.readouterr().err


def test_cli_warns_when_references_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without references the SHARK term silently cannot fire, so say so."""
    path = tmp_path / "store.json"
    write_domain_store(_labelled_store(), path)
    monkeypatch.setattr(validation_cli, "default_reference_store", lambda: tmp_path / "absent.json")

    validate_main(
        [
            str(path),
            "-o",
            str(tmp_path / "report.json"),
            "--permutations",
            "3",
            "--recurrence-permutations",
            "3",
            "--disorder-backend",
            BACKEND_FOLDINDEX,
            "--shark-backend",
            BACKEND_KMER,
        ],
    )
    assert "bundled reference JDPs not found" in capsys.readouterr().err
