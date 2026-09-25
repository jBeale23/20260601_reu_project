"""Tests for the Poisson-Boltzmann solvation analysis."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from validation.electrostatics import (
    MIN_GROUP,
    SolvationEnergy,
    compare_groups,
    effect_label,
    electrostatics_report,
    load_energies,
    stratified_by_covariate,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, name: str, rows: list[str]) -> Path:
    """Write a tab-separated fixture file and return its path."""
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n")
    return path


def test_load_energies_reads_scraper_rows(tmp_path: Path) -> None:
    """Accession, solvation, and reference come back as written."""
    path = _write(tmp_path, "e.tsv", ["A1\t-100.5\t2000.0", "A2\t-200.25\t3000.5"])
    energies = load_energies([path])
    assert set(energies) == {"A1", "A2"}
    assert energies["A1"].solvation == pytest.approx(-100.5)
    assert energies["A2"].reference == pytest.approx(3000.5)


def test_load_energies_skips_header_and_malformed_rows(tmp_path: Path) -> None:
    """A header or an unparseable row is dropped, never defaulted to zero."""
    path = _write(
        tmp_path,
        "e.tsv",
        ["Protein\tsolv\tref", "A1\t-100.0\t2000.0", "A2\tnot-a-number\t3000.0", "A3\t-50.0"],
    )
    energies = load_energies([path])
    # A malformed row must be dropped, never defaulted to zero: a zero energy would drag
    # every group median toward it and read as a real result.
    assert set(energies) == {"A1"}


def test_load_energies_rejects_non_finite(tmp_path: Path) -> None:
    """NaN and infinity are not energies and must not enter a median."""
    path = _write(tmp_path, "e.tsv", ["A1\tnan\t2000.0", "A2\tinf\t2000.0", "A3\t-10.0\t20.0"])
    assert set(load_energies([path])) == {"A3"}


def test_load_energies_merges_multiple_files(tmp_path: Path) -> None:
    """The run writes one file per array task, so loading must combine them."""
    first = _write(tmp_path, "a.tsv", ["A1\t-1.0\t2.0"])
    second = _write(tmp_path, "b.tsv", ["A2\t-3.0\t4.0"])
    assert set(load_energies([first, second])) == {"A1", "A2"}


def test_per_residue_guards_zero_length() -> None:
    """A zero or negative length yields None rather than a division error."""
    energy = SolvationEnergy("A1", -100.0, 200.0)
    assert energy.per_residue(0) is None
    assert energy.per_residue(-5) is None
    assert energy.per_residue(10) == pytest.approx(-10.0)


def test_effect_label_thresholds() -> None:
    """Cliff's delta is named by the conventional cut points, sign-independently."""
    assert effect_label(0.10) == "negligible"
    assert effect_label(0.20) == "small"
    assert effect_label(0.40) == "medium"
    assert effect_label(0.90) == "large"
    assert effect_label(-0.90) == "large"


def test_compare_groups_skips_undersized_groups() -> None:
    """A group too small to have a stable median is not tested at all."""
    values = {f"A{i}": float(i) for i in range(100)}
    groups = {f"A{i}": ("big" if i >= 5 else "tiny") for i in range(100)}
    results = compare_groups(values, groups, "m")
    assert [r.group for r in results] == []  # "tiny" too small, so "big" has no comparator


def test_compare_groups_detects_a_real_shift() -> None:
    """Fully separated groups give delta of one, in the right direction."""
    values = {}
    groups = {}
    for i in range(60):
        values[f"L{i}"] = float(i)
        groups[f"L{i}"] = "low"
    for i in range(60):
        values[f"H{i}"] = float(i) + 1000.0
        groups[f"H{i}"] = "high"
    results = {r.group: r for r in compare_groups(values, groups, "m")}
    assert results["high"].delta == pytest.approx(1.0)
    assert results["low"].delta == pytest.approx(-1.0)
    assert results["high"].significant


def test_significance_requires_effect_not_just_p_value() -> None:
    """At this n any difference clears a p-value; only the effect gate stops noise."""
    # Two groups that differ by a hair, at a size where any p-value threshold is trivially
    # cleared. The effect gate is what stops this being reported as a finding.
    values = {}
    groups = {}
    for i in range(400):
        values[f"A{i}"] = float(i)
        groups[f"A{i}"] = "a"
        values[f"B{i}"] = float(i) + 0.001
        groups[f"B{i}"] = "b"
    for result in compare_groups(values, groups, "m"):
        assert abs(result.delta) < 0.147
        assert not result.significant


def test_report_flags_length_as_the_confound() -> None:
    """Energy proportional to length must survive raw and die per residue."""
    # Energy is exactly proportional to length, and the classes differ only in length.
    # The raw comparison must therefore find a difference and the per-residue one must not.
    energies, lengths, groups = {}, {}, {}
    for i in range(MIN_GROUP * 3):
        short, long_ = f"S{i}", f"L{i}"
        lengths[short], lengths[long_] = 100, 400
        energies[short] = SolvationEnergy(short, -100.0 * 10, 0.0)
        energies[long_] = SolvationEnergy(long_, -400.0 * 10, 0.0)
        groups[short], groups[long_] = "small_class", "large_class"

    report = electrostatics_report(energies, lengths, groups)
    raw = {r["group"]: r for r in report["raw"]}
    norm = {r["group"]: r for r in report["per_residue"]}

    assert raw["large_class"]["significant"] is True
    assert norm["large_class"]["significant"] is False
    assert report["n_significant_per_residue"] == 0
    assert "size effect" in report["interpretation"]


def test_report_keeps_a_genuine_per_residue_difference() -> None:
    """Equal lengths, different energy density: this is the real signal."""
    # Same lengths in both classes, different energy density: this must survive.
    energies, lengths, groups = {}, {}, {}
    for i in range(MIN_GROUP * 3):
        a, b = f"A{i}", f"B{i}"
        lengths[a] = lengths[b] = 200
        energies[a] = SolvationEnergy(a, -200.0 * 5 + i, 0.0)
        energies[b] = SolvationEnergy(b, -200.0 * 50 + i, 0.0)
        groups[a], groups[b] = "sparse", "dense"

    report = electrostatics_report(energies, lengths, groups)
    norm = {r["group"]: r for r in report["per_residue"]}
    assert norm["dense"]["significant"] is True
    assert abs(norm["dense"]["cliffs_delta"]) > 0.474


def test_report_counts_proteins_without_length() -> None:
    """Proteins lacking a length are excluded and counted, not silently dropped."""
    energies = {f"A{i}": SolvationEnergy(f"A{i}", -1.0 * i, 0.0) for i in range(10)}
    lengths = {f"A{i}": 100 for i in range(4)}
    report = electrostatics_report(energies, lengths, {})
    assert report["n_energies_loaded"] == 10
    assert report["n_proteins"] == 4
    assert report["n_without_length"] == 6


def test_report_is_json_safe() -> None:
    """The report must serialise, with finite correlation values."""
    energies = {f"A{i}": SolvationEnergy(f"A{i}", -float(i), 1.0) for i in range(MIN_GROUP * 2)}
    lengths = dict.fromkeys(energies, 100)
    groups = {a: ("x" if index % 2 else "y") for index, a in enumerate(energies)}
    report = electrostatics_report(energies, lengths, groups)
    for value in report["size_confound"].values():
        assert math.isfinite(value)
    json.dumps(report)


def test_stratification_unmasks_a_covariate_proxy() -> None:
    """A class that only differs because it is disordered must not look consistent."""
    # Energy is a pure function of the covariate; class membership is assigned by the
    # covariate too. Within a stratum there is then nothing left to find.
    values, groups, covariate = {}, {}, {}
    for i in range(400):
        accession = f"A{i}"
        d = i / 400.0
        covariate[accession] = d
        values[accession] = -100.0 * d
        groups[accession] = "disordered" if d > 0.5 else "ordered"
    result = stratified_by_covariate(values, groups, covariate, "m")
    assert result["consistent_groups"] == []


def test_stratification_keeps_an_effect_independent_of_the_covariate() -> None:
    """A class offset that holds at every covariate level is not a proxy."""
    values, groups, covariate = {}, {}, {}
    for i in range(400):
        d = (i % 100) / 100.0
        for label, offset in (("high", -50.0), ("low", 0.0)):
            accession = f"{label}{i}"
            covariate[accession] = d
            # Same covariate dependence in both classes, plus a constant class offset.
            values[accession] = -100.0 * d + offset + (i % 7) * 0.01
            groups[accession] = label
    result = stratified_by_covariate(values, groups, covariate, "m")
    assert set(result["consistent_groups"]) == {"high", "low"}
    assert result["by_group"]["high"]["sign_stable"] is True


def test_stratification_needs_enough_data() -> None:
    """Too few proteins to fill the strata yields an empty result, not a spurious one."""
    values = {f"A{i}": float(i) for i in range(10)}
    groups = dict.fromkeys(values, "x")
    covariate = {a: float(i) for i, a in enumerate(values)}
    result = stratified_by_covariate(values, groups, covariate, "m")
    assert result["strata"] == []
    assert result["by_group"] == {}
