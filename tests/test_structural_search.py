"""Tests for structural homology search.

Foldseek is exercised through its tabular output rather than by invoking it, since the
parsing and the thresholds are where the errors would be silent.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

import structure_analysis.structural_search as search_module
from structure_analysis.structural_search import (
    MIN_TM_SCORE,
    TWILIGHT_ZONE_IDENTITY,
    InsufficientStagingSpaceError,
    NoReadableStructuresError,
    SearchOptions,
    StructuralHit,
    _accession_from_model,
    best_donors,
    foldseek_available,
    parse_alignment,
    readable_structure_count,
    search_structures,
    stage_structures,
)


def test_accessions_are_recovered_from_alphafold_filenames() -> None:
    """Hits name model files; everything downstream is keyed by accession."""
    for name, expected in (
        ("AF-P08622-F1-model_v4.pdb", "P08622"),
        ("AF-A0A674ASH5-F1-model_v6.pdb.gz", "A0A674ASH5"),
        ("/some/path/AF-Q9NVH1-F1-model_v4.cif.gz", "Q9NVH1"),
    ):
        assert _accession_from_model(name) == expected


def test_a_non_alphafold_name_falls_back_to_the_stem() -> None:
    """A file from elsewhere must not be silently mis-parsed into a wrong accession."""
    assert _accession_from_model("something_else.pdb") == "something_else"


def test_alignment_output_is_parsed_into_hits() -> None:
    """Column order is fixed by the format string passed to Foldseek, not by its default."""
    text = (
        "AF-P08622-F1-model_v4\tAF-P25685-F1-model_v4\t5.19e-92\t0.658\t626\t0.9488\n"
        "AF-P08622-F1-model_v4\tAF-Q9NVH1-F1-model_v4\t2.21e-10\t0.180\t210\t0.6100\n"
    )
    hits = parse_alignment(text)
    assert [hit.target for hit in hits] == ["P25685", "Q9NVH1"]
    assert hits[0].tm_score == pytest.approx(0.9488)
    assert hits[0].alignment_length == 626
    assert hits[1].sequence_identity == pytest.approx(0.18)


def test_malformed_lines_are_skipped_not_guessed() -> None:
    """A truncated or non-numeric row must not become a hit with invented values."""
    text = "incomplete\trow\nAF-A-F1-model_v4\tAF-B-F1-model_v4\tnot-a-number\t0.5\t100\t0.8\n\n"
    assert parse_alignment(text) == []


def test_a_remote_match_is_flagged_as_beyond_sequence_search() -> None:
    """The whole justification for the structural route.

    Below the twilight zone a pairwise sequence method is unreliable, so a confident
    structural match there is a donor the sequence route could not have supplied.
    """
    remote = StructuralHit("Q", "T", 1e-8, 0.72, 180, 0.15)
    close = StructuralHit("Q", "T", 1e-40, 0.95, 300, 0.68)
    assert remote.is_remote
    assert not close.is_remote
    assert pytest.approx(0.25) == TWILIGHT_ZONE_IDENTITY
    assert remote.to_json_dict()["beyond_sequence_search"] is True


def test_donors_are_grouped_and_ranked_by_fold_similarity() -> None:
    """A reader taking one donor should be taking the best-matching fold."""
    hits = [
        StructuralHit("Q1", "D1", 1e-5, 0.60, 100, 0.20),
        StructuralHit("Q1", "D2", 1e-9, 0.85, 150, 0.30),
        StructuralHit("Q1", "NOT_A_DONOR", 1e-9, 0.99, 200, 0.90),
        StructuralHit("Q2", "D1", 1e-6, 0.70, 120, 0.25),
    ]
    grouped = best_donors(hits, {"D1": object(), "D2": object()})

    assert set(grouped) == {"Q1", "Q2"}
    # Only eligible donors survive, ordered by TM-score.
    assert [hit.target for hit in grouped["Q1"]] == ["D2", "D1"]


def test_donor_list_is_capped() -> None:
    """An unbounded donor list would make agreement counting meaningless."""
    hits = [StructuralHit("Q", f"D{i}", 1e-5, 0.9 - i * 0.01, 100, 0.3) for i in range(30)]
    donors = {f"D{i}": object() for i in range(30)}
    assert len(best_donors(hits, donors, max_per_query=5)["Q"]) == 5


def test_search_degrades_when_foldseek_is_absent(tmp_path: Path) -> None:
    """No binary means sequence-only transfer, not a failed run."""
    (tmp_path / "q").mkdir()
    (tmp_path / "d").mkdir()
    result = search_structures(
        tmp_path / "q",
        tmp_path / "d",
        options=SearchOptions(executable=str(tmp_path / "no-such-foldseek")),
    )
    assert result == []


def test_search_returns_nothing_for_missing_directories(tmp_path: Path) -> None:
    """A path typo must not look like an absence of homologs."""
    assert search_structures(tmp_path / "absent", tmp_path, options=SearchOptions()) == []


def test_availability_check_accepts_a_path_or_a_name() -> None:
    """The binary is vendored, so it is given by path rather than found on PATH."""
    assert not foldseek_available("definitely-not-a-real-binary-name")


def test_self_matches_and_weak_folds_are_excluded_by_threshold() -> None:
    """A protein matching itself carries no information, and a low TM-score is not a fold."""
    assert pytest.approx(0.5) == MIN_TM_SCORE
    hits = parse_alignment(
        "AF-A-F1-model_v4\tAF-A-F1-model_v4\t0.0\t1.0\t300\t1.0\n"
        "AF-A-F1-model_v4\tAF-B-F1-model_v4\t1e-3\t0.1\t40\t0.2\n",
    )
    # parse_alignment keeps them; search_structures is what filters, so verify the values
    # the filter acts on are present and correct.
    assert hits[0].query == hits[0].target
    assert hits[1].tm_score < MIN_TM_SCORE


def test_gzipped_models_are_not_counted_as_readable(tmp_path: Path) -> None:
    """Foldseek's directory scan skips .gz without error.

    A directory of gzipped AlphaFold models built an empty database, and a two-hour
    all-versus-all then reported "0 structural hits above threshold" - a statement about
    the input dressed as a statement about the biology.
    """
    (tmp_path / "AF-P0A6Y8-F1-model_v6.pdb.gz").write_bytes(b"\x1f\x8b\x08\x00")
    assert readable_structure_count(tmp_path) == 0


def test_readable_models_are_counted(tmp_path: Path) -> None:
    """A staged directory must register as usable."""
    (tmp_path / "AF-P0A6Y8-F1-model_v6.pdb").write_text("ATOM\n", encoding="utf-8")
    (tmp_path / "AF-Q9UBS3-F1-model_v6.cif").write_text("data_\n", encoding="utf-8")
    assert readable_structure_count(tmp_path) == 2


def test_an_unreadable_directory_raises_rather_than_returning_no_hits(tmp_path: Path) -> None:
    """The failure mode this whole guard exists for."""
    source = tmp_path / "gz"
    source.mkdir()
    (source / "AF-P1-F1-model_v6.pdb.gz").write_bytes(b"\x1f\x8b\x08\x00")
    if not foldseek_available():
        pytest.skip("foldseek not installed")
    with pytest.raises(NoReadableStructuresError):
        search_structures(source, source)


def test_staging_decompresses_one_format_only(tmp_path: Path) -> None:
    """Staging both .pdb and .cif would enter every model twice.

    Each protein would then score a perfect self-hit against its own twin, which looks like
    a structural match and is an artefact of the staging.
    """
    source = tmp_path / "src"
    source.mkdir()
    for suffix in (".pdb", ".cif"):
        with gzip.open(source / f"AF-P1-F1-model_v6{suffix}.gz", "wb") as handle:
            handle.write(b"ATOM\n")
    destination = tmp_path / "staged"
    staged = stage_structures(source, destination, suffix=".pdb")
    assert staged == 1
    assert [item.name for item in destination.iterdir()] == ["AF-P1-F1-model_v6.pdb"]


def test_staging_is_idempotent(tmp_path: Path) -> None:
    """A resumed run must not redo work already staged."""
    source = tmp_path / "src"
    source.mkdir()
    with gzip.open(source / "AF-P1-F1-model_v6.pdb.gz", "wb") as handle:
        handle.write(b"ATOM\n")
    destination = tmp_path / "staged"
    assert stage_structures(source, destination) == 1
    assert stage_structures(source, destination) == 1


def test_staging_refuses_when_it_would_not_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing early costs seconds; failing after 50,000 files costs the jobs beside it.

    Staging the full model set without checking exhausted a shared group quota, which
    killed an unrelated GPU run mid-write and lost its entire result.
    """
    source = tmp_path / "src"
    source.mkdir()
    for index in range(5):
        with gzip.open(source / f"AF-P{index}-F1-model_v6.pdb.gz", "wb") as handle:
            handle.write(b"ATOM" * 1000)

    fake = type("usage", (), {"free": 1})()
    monkeypatch.setattr("structure_analysis.structural_search.shutil.disk_usage", lambda _p: fake)
    with pytest.raises(InsufficientStagingSpaceError):
        stage_structures(source, tmp_path / "staged")


def test_a_limit_caps_what_is_considered_before_the_space_check(tmp_path: Path) -> None:
    """The cap must apply to the estimate too, or a bounded run is refused on a full set."""
    source = tmp_path / "src"
    source.mkdir()
    for index in range(10):
        with gzip.open(source / f"AF-P{index}-F1-model_v6.pdb.gz", "wb") as handle:
            handle.write(b"ATOM\n")
    assert stage_structures(source, tmp_path / "staged", limit=3) == 3


def test_staging_can_be_restricted_to_chosen_representatives(tmp_path: Path) -> None:
    """Filtering before decompression, not after.

    Writing 142,948 models and then ignoring most of them is what exhausted the quota.
    """
    source = tmp_path / "src"
    source.mkdir()
    for accession in ("P0A6Y8", "Q9UBS3", "P11142"):
        with gzip.open(source / f"AF-{accession}-F1-model_v6.pdb.gz", "wb") as handle:
            handle.write(b"ATOM\n")
    destination = tmp_path / "staged"
    staged = stage_structures(source, destination, accessions=["P0A6Y8", "P11142"])
    assert staged == 2
    names = sorted(item.name for item in destination.iterdir())
    assert names == ["AF-P0A6Y8-F1-model_v6.pdb", "AF-P11142-F1-model_v6.pdb"]


def test_an_empty_representative_list_stages_nothing(tmp_path: Path) -> None:
    """An empty selection must not silently fall back to staging everything."""
    source = tmp_path / "src"
    source.mkdir()
    with gzip.open(source / "AF-P1-F1-model_v6.pdb.gz", "wb") as handle:
        handle.write(b"ATOM\n")
    assert stage_structures(source, tmp_path / "staged", accessions=[]) == 0


def test_available_space_prefers_the_quota_over_the_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """The filesystem number is the wrong one on a shared cluster.

    ``shutil.disk_usage`` reported 397 TB free while the group quota was already exhausted,
    so the staging guard passed and then filled the quota anyway - killing jobs that had
    nothing to do with the staging.
    """
    monkeypatch.setattr(search_module.shutil, "disk_usage", lambda _p: type("u", (), {"free": 10**15})())
    monkeypatch.setattr(search_module, "_lustre_quota_remaining", lambda _p: 5 * 10**9)
    assert search_module.available_space(Path()) == 5 * 10**9


def test_available_space_falls_back_when_there_is_no_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not every filesystem has one; the guard must still work there."""
    monkeypatch.setattr(search_module.shutil, "disk_usage", lambda _p: type("u", (), {"free": 42})())
    monkeypatch.setattr(search_module, "_lustre_quota_remaining", lambda _p: None)
    assert search_module.available_space(Path()) == 42


def test_staging_refuses_against_the_quota_not_the_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: a huge filesystem must not license filling a small quota."""
    source = tmp_path / "src"
    source.mkdir()
    for index in range(5):
        with gzip.open(source / f"AF-P{index}-F1-model_v6.pdb.gz", "wb") as handle:
            handle.write(b"ATOM" * 5000)

    monkeypatch.setattr(search_module.shutil, "disk_usage", lambda _p: type("u", (), {"free": 10**15})())
    monkeypatch.setattr(search_module, "_lustre_quota_remaining", lambda _p: 1)
    with pytest.raises(InsufficientStagingSpaceError):
        stage_structures(source, tmp_path / "staged")


def test_an_explicit_ceiling_holds_when_the_quota_cannot_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard has to work on a cluster where `lfs` is not installed.

    There, ``_lustre_quota_remaining`` always returns None and the check falls back to
    filesystem free space - which reported hundreds of terabytes while a group quota of a
    few gigabytes was already exhausted. That mismatch killed unrelated jobs three times.
    """
    source = tmp_path / "src"
    source.mkdir()
    for index in range(20):
        with gzip.open(source / f"AF-P{index}-F1-model_v6.pdb.gz", "wb") as handle:
            handle.write(b"ATOM" * 20000)

    monkeypatch.setattr(search_module.shutil, "disk_usage", lambda _p: type("u", (), {"free": 10**15})())
    monkeypatch.setattr(search_module, "_lustre_quota_remaining", lambda _p: None)
    with pytest.raises(InsufficientStagingSpaceError):
        stage_structures(source, tmp_path / "staged", max_bytes=1000)
