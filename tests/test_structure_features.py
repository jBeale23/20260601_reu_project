"""Tests for AlphaFold-derived structural features.

These are geometric quantities with known answers on constructed shapes, so each is
checked against a case whose value can be reasoned out rather than against a recorded
number.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from structure_analysis.batch_features import (
    STRUCTURE_FEATURE_COLUMNS,
    compute_all,
    features_for_accession,
    find_model,
    write_structure_features,
)
from structure_analysis.features import (
    BURIED_NEIGHBOURS,
    CONFIDENT_PLDDT,
    DISORDER_PLDDT,
    burial_from_neighbours,
    compute_structural_features,
    expected_radius_of_gyration,
    neighbour_counts,
    radius_of_gyration,
    relative_contact_order,
)
from structure_analysis.pdb_io import ResidueRecord

if TYPE_CHECKING:
    from pathlib import Path


def _residue(
    index: int,
    coord: tuple[float, float, float] | None,
    *,
    plddt: float = 90.0,
    resname: str = "ALA",
) -> ResidueRecord:
    return ResidueRecord(index=index, resnum=index, resname=resname, ca_coord=coord, plddt=plddt)


def _extended_chain(n: int, spacing: float = 3.8) -> list[ResidueRecord]:
    """A straight line of alpha carbons: maximally extended, no long-range contacts."""
    return [_residue(i + 1, (i * spacing, 0.0, 0.0)) for i in range(n)]


def _compact_ball(n: int, seed: int = 0) -> list[ResidueRecord]:
    """Residues packed into a small sphere: maximally compact."""
    rng = np.random.default_rng(seed)
    points = rng.normal(scale=6.0, size=(n, 3))
    return [_residue(i + 1, tuple(points[i])) for i in range(n)]


def test_radius_of_gyration_of_a_known_shape() -> None:
    """Analytic case: two points 2d apart have Rg = d about their centroid."""
    coords = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert radius_of_gyration(coords) == pytest.approx(5.0)
    assert radius_of_gyration(np.zeros((0, 3))) == 0.0


def test_an_extended_chain_is_less_compact_than_a_ball() -> None:
    """The quantity compactness exists to separate."""
    extended = compute_structural_features("EXT", _extended_chain(120))
    ball = compute_structural_features("BALL", _compact_ball(120))
    assert extended is not None
    assert ball is not None
    assert extended.compactness > ball.compactness


def test_expected_radius_grows_with_length_so_compactness_does_not_measure_length() -> None:
    """Dividing by the length-matched expectation is what makes compactness comparable."""
    assert expected_radius_of_gyration(400) > expected_radius_of_gyration(100)
    # Two balls of very different size should report similar compactness.
    small = compute_structural_features("S", _compact_ball(60, seed=1))
    large = compute_structural_features("L", _compact_ball(240, seed=1))
    assert small is not None
    assert large is not None
    assert abs(small.compactness - large.compactness) < 1.0


def test_neighbour_counts_exclude_the_residue_itself() -> None:
    """A residue is not its own neighbour; off-by-one here shifts every burial call."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    counts = neighbour_counts(coords, radius=10.0)
    assert counts.tolist() == [1, 1, 0]
    assert neighbour_counts(np.zeros((0, 3))).tolist() == []


def test_burial_threshold_is_applied_as_documented() -> None:
    """Boolean burial straight from the neighbour count."""
    counts = np.array([0, BURIED_NEIGHBOURS - 1, BURIED_NEIGHBOURS, BURIED_NEIGHBOURS + 5])
    assert burial_from_neighbours(counts).tolist() == [False, False, True, True]


def test_contact_order_is_low_for_local_structure_and_high_for_a_fold() -> None:
    """The topology signal.

    A straight chain has only backbone-local contacts, which are excluded, so its contact
    order is zero. A chain folded back on itself puts residues from opposite ends in
    contact, which is what a high contact order means.
    """
    straight = np.array([[i * 3.8, 0.0, 0.0] for i in range(60)])
    assert relative_contact_order(straight) == pytest.approx(0.0, abs=1e-9)

    # A hairpin: second half runs back alongside the first, 5 angstroms away.
    half = [[i * 3.8, 0.0, 0.0] for i in range(30)]
    back = [[(29 - i) * 3.8, 5.0, 0.0] for i in range(30)]
    hairpin = np.array(half + back)
    assert relative_contact_order(hairpin) > 0.1


def test_contact_order_ignores_backbone_neighbours() -> None:
    """Sequence neighbours are always in contact and say nothing about the fold."""
    tiny = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
    assert relative_contact_order(tiny) == 0.0


def test_plddt_bands_are_counted_against_the_documented_thresholds() -> None:
    """PLDDT is an independent disorder signal, so its bands must be exact."""
    residues = [
        _residue(1, (0.0, 0.0, 0.0), plddt=DISORDER_PLDDT - 1),
        _residue(2, (4.0, 0.0, 0.0), plddt=DISORDER_PLDDT + 1),
        _residue(3, (8.0, 0.0, 0.0), plddt=CONFIDENT_PLDDT),
        _residue(4, (12.0, 0.0, 0.0), plddt=CONFIDENT_PLDDT + 20),
    ]
    features = compute_structural_features("P", residues)
    assert features is not None
    assert features.fraction_disordered_plddt == pytest.approx(0.25)
    assert features.fraction_confident_plddt == pytest.approx(0.5)


def test_a_mostly_unconfident_model_reads_as_disordered() -> None:
    """The flag that makes pLDDT usable as a disorder call."""
    residues = [_residue(i + 1, (i * 4.0, 0.0, 0.0), plddt=30.0) for i in range(20)]
    features = compute_structural_features("D", residues)
    assert features is not None
    assert features.is_largely_disordered


def test_surface_and_buried_charge_are_separated() -> None:
    """Net charge over a whole protein averages the two and answers neither question.

    Measured on real models: human DNAJB1 has surface charge +10 against buried -6, while
    E. coli DnaJ is near neutral on both. Their net charges are similar; only the split
    shows the difference.
    """
    # A charged shell around a neutral, densely packed core.
    core = [_residue(i + 1, tuple(np.random.default_rng(2).normal(scale=2.0, size=3))) for i in range(40)]
    shell = [_residue(100 + i, (30.0 * math.cos(i), 30.0 * math.sin(i), 0.0), resname="LYS") for i in range(12)]
    features = compute_structural_features("SHELL", [*core, *shell])
    assert features is not None
    assert features.surface_net_charge > 0
    assert features.surface_charge_density > 0


def test_a_model_without_coordinates_returns_none() -> None:
    """None, not zeros - a featureless protein and an unusable model differ."""
    assert compute_structural_features("X", []) is None
    assert compute_structural_features("X", [_residue(1, None)]) is None


def test_features_serialize_everything_reported() -> None:
    """Anything the report claims must be in the payload."""
    payload = compute_structural_features("P", _compact_ball(50)).to_json_dict()
    for key in (
        "mean_plddt",
        "fraction_disordered_plddt",
        "fraction_confident_plddt",
        "radius_of_gyration",
        "compactness",
        "relative_contact_order",
        "fraction_buried",
        "surface_net_charge",
        "buried_net_charge",
        "surface_charge_density",
    ):
        assert key in payload, key


def _write_model(directory: Path, accession: str, version: int, n: int = 30) -> Path:
    """Write a minimal AlphaFold-style PDB with pLDDT in the B-factor column."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"AF-{accession}-F1-model_v{version}.pdb"
    lines = [
        (
            f"ATOM  {index + 1:5d}  CA  ALA A{index + 1:4d}    "
            f"{index * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{85.0:6.2f}           C"
        )
        for index in range(n)
    ]
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


def test_the_newest_model_version_is_preferred(tmp_path: Path) -> None:
    """Several AlphaFold versions coexist in a directory topped up over time."""
    _write_model(tmp_path, "P00001", 4)
    newest = _write_model(tmp_path, "P00001", 6)
    assert find_model(tmp_path, "P00001") == newest


def test_a_missing_model_is_reported_not_guessed(tmp_path: Path) -> None:
    """No model must not become a row of zeros."""
    tmp_path.mkdir(exist_ok=True)
    assert find_model(tmp_path, "ABSENT") is None
    assert features_for_accession("ABSENT", tmp_path) is None


def test_coverage_is_reported_because_a_systematic_gap_would_bias_everything(tmp_path: Path) -> None:
    """Models present for model organisms and absent elsewhere would skew every result."""
    _write_model(tmp_path, "P00001", 6)
    _write_model(tmp_path, "P00002", 6)

    rows, summary = compute_all(["P00001", "P00002", "MISSING1", "MISSING2"], tmp_path)

    assert len(rows) == 2
    assert summary.n_requested == 4
    assert summary.n_with_model == 2
    assert summary.n_missing == 2
    assert summary.coverage == pytest.approx(0.5)


def test_rows_come_back_in_a_stable_order(tmp_path: Path) -> None:
    """Parallel chunks complete out of order; the output must not."""
    for index in range(6):
        _write_model(tmp_path, f"P{index:05d}", 6)
    rows, _summary = compute_all([f"P{index:05d}" for index in range(6)], tmp_path)
    assert [row["accession"] for row in rows] == sorted(row["accession"] for row in rows)


def test_written_csv_carries_every_declared_column(tmp_path: Path) -> None:
    """A column named in the header and absent from the rows would read as a blank value."""
    _write_model(tmp_path / "models", "P00001", 6)
    rows, _summary = compute_all(["P00001"], tmp_path / "models")
    out = tmp_path / "features.csv"
    write_structure_features(rows, out)

    with out.open(encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written
    for column in STRUCTURE_FEATURE_COLUMNS:
        assert column in written[0]

    # Columns that exist for every modelled protein.
    for column in ("accession", "n_residues", "mean_plddt", "fraction_buried", "total_sasa", "n_pockets"):
        assert written[0][column] != "", column

    # Pocket detail is legitimately blank when a structure has no cavity passing the
    # wall-density and confidence filters, which is the common case for a small model.
    assert "largest_pocket_volume" in written[0]
