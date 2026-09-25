"""Tests for solvent accessibility and pocket geometry.

Geometric quantities with analytically known answers on constructed shapes, so each is
checked against a value that can be reasoned out rather than a recorded number.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from structure_analysis.surface import (
    EXPOSED_RSA,
    MIN_LINING_DENSITY,
    MIN_POCKET_PLDDT,
    PROBE_RADIUS,
    Atom,
    atom_sasa,
    find_pockets,
    load_atoms,
    residue_surfaces,
)

if TYPE_CHECKING:
    from pathlib import Path


def _atom(resnum: int, resname: str, name: str, coord: tuple[float, float, float], plddt: float = 90.0) -> Atom:
    return Atom(resnum=resnum, resname=resname, name=name, element=name[0], coord=coord, plddt=plddt)


def test_an_isolated_atom_has_the_full_sphere_area() -> None:
    """Analytic case: nothing occludes it, so area = 4*pi*(r + probe)^2."""
    atom = _atom(1, "ALA", "CA", (0.0, 0.0, 0.0))
    area = atom_sasa([atom])[0]
    expected = 4.0 * math.pi * (atom.radius + PROBE_RADIUS) ** 2
    assert area == pytest.approx(expected, rel=0.02)


def test_burying_an_atom_reduces_its_area_to_zero() -> None:
    """An atom surrounded on all sides presents no surface to solvent."""
    centre = _atom(1, "ALA", "CA", (0.0, 0.0, 0.0))
    shell = [
        _atom(2 + i, "ALA", "CB", tuple(2.4 * np.array(direction)))
        for i, direction in enumerate(
            [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
                (1, 1, 1),
                (-1, -1, -1),
                (1, -1, 1),
                (-1, 1, -1),
                (1, 1, -1),
                (-1, -1, 1),
            ],
        )
    ]
    areas = atom_sasa([centre, *shell])
    assert areas[0] < 0.05 * (4.0 * math.pi * (centre.radius + PROBE_RADIUS) ** 2)


def test_two_touching_atoms_each_lose_area_but_not_all_of_it() -> None:
    """Partial occlusion is partial, which a boolean buried/exposed test would miss."""
    pair = [_atom(1, "ALA", "CA", (0.0, 0.0, 0.0)), _atom(2, "ALA", "CB", (3.0, 0.0, 0.0))]
    areas = atom_sasa(pair)
    full = 4.0 * math.pi * (pair[0].radius + PROBE_RADIUS) ** 2
    for area in areas:
        assert 0.3 * full < area < full


def test_relative_accessibility_normalises_for_residue_size() -> None:
    """Relative accessibility is what means "exposed".

    Absolute area scales with residue size, so a fully exposed glycine and a buried
    tryptophan can report the same square angstroms.
    """
    glycine = _atom(1, "GLY", "CA", (0.0, 0.0, 0.0))
    tryptophan = _atom(2, "TRP", "CA", (60.0, 0.0, 0.0))
    atoms = [glycine, tryptophan]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))

    by_name = {item.resname: item for item in surfaces}
    # Both isolated, so both are fully exposed however different their absolute areas are.
    assert by_name["GLY"].relative_sasa > EXPOSED_RSA
    assert by_name["TRP"].relative_sasa > EXPOSED_RSA
    assert by_name["GLY"].absolute_sasa == pytest.approx(by_name["TRP"].absolute_sasa, rel=0.05)
    # Normalisation makes glycine the more exposed of the two despite equal absolute area.
    assert by_name["GLY"].relative_sasa > by_name["TRP"].relative_sasa


def test_residue_charge_is_assigned_at_physiological_ph() -> None:
    """His is left neutral, matching the sequence-side charge convention."""
    atoms = [
        _atom(1, "LYS", "CA", (0.0, 0.0, 0.0)),
        _atom(2, "ASP", "CA", (20.0, 0.0, 0.0)),
        _atom(3, "HIS", "CA", (40.0, 0.0, 0.0)),
    ]
    charges = {item.resname: item.charge for item in residue_surfaces(atoms, atom_sasa(atoms))}
    assert charges == {"LYS": 1.0, "ASP": -1.0, "HIS": 0.0}


def test_a_solid_ball_has_no_internal_pocket() -> None:
    """No cavity means no pocket; a detector that finds one everywhere is useless."""
    rng = np.random.default_rng(0)
    points = rng.normal(scale=5.0, size=(220, 3))
    atoms = [_atom(i + 1, "ALA", "CA", tuple(points[i])) for i in range(len(points))]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))
    assert find_pockets(atoms, surfaces) == []


def test_inter_domain_space_is_rejected_as_a_pocket() -> None:
    """The failure the wall-density filter exists to prevent.

    Two domains held apart leave a gap enclosed on every scan axis, so it passes the
    geometric test. It is not a pocket: it has almost no walls for its size. The first
    validation run reported exactly this as a 1,675-cubic-angstrom cavity lined by eleven
    residues.
    """
    rng = np.random.default_rng(1)
    left = rng.normal(loc=(-14, 0, 0), scale=4.0, size=(120, 3))
    right = rng.normal(loc=(14, 0, 0), scale=4.0, size=(120, 3))
    points = np.vstack([left, right])
    atoms = [_atom(i + 1, "ALA", "CA", tuple(points[i])) for i in range(len(points))]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))

    for pocket in find_pockets(atoms, surfaces):
        density = pocket.n_lining_residues / (pocket.volume / 100.0)
        assert density >= MIN_LINING_DENSITY


def test_low_confidence_cavities_are_discarded() -> None:
    """A cavity whose surroundings AlphaFold could not place is an artefact of that."""
    rng = np.random.default_rng(2)
    left = rng.normal(loc=(-13, 0, 0), scale=4.0, size=(120, 3))
    right = rng.normal(loc=(13, 0, 0), scale=4.0, size=(120, 3))
    points = np.vstack([left, right])
    atoms = [_atom(i + 1, "ALA", "CA", tuple(points[i]), plddt=20.0) for i in range(len(points))]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))

    for pocket in find_pockets(atoms, surfaces):
        assert pocket.mean_plddt is None or pocket.mean_plddt >= MIN_POCKET_PLDDT


def test_pockets_are_ranked_largest_first() -> None:
    """A reader looking at one pocket should be looking at the biggest."""
    rng = np.random.default_rng(3)
    points = rng.normal(scale=9.0, size=(400, 3))
    atoms = [_atom(i + 1, "ALA", "CA", tuple(points[i])) for i in range(len(points))]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))
    pockets = find_pockets(atoms, surfaces)
    volumes = [pocket.volume for pocket in pockets]
    assert volumes == sorted(volumes, reverse=True)
    assert [pocket.rank for pocket in pockets] == list(range(1, len(pockets) + 1))


def test_pocket_serializes_charge_and_accessibility_together() -> None:
    """The combination is what distinguishes a binding site from a folding core."""
    rng = np.random.default_rng(4)
    points = rng.normal(scale=9.0, size=(400, 3))
    atoms = [_atom(i + 1, "LYS", "CA", tuple(points[i])) for i in range(len(points))]
    surfaces = residue_surfaces(atoms, atom_sasa(atoms))
    pockets = find_pockets(atoms, surfaces)
    if not pockets:
        pytest.skip("no cavity in this random packing")
    payload = pockets[0].to_json_dict()
    for key in ("volume", "depth", "net_charge", "mean_relative_sasa", "fraction_exposed_lining", "mean_plddt"):
        assert key in payload, key


def test_atom_reader_skips_hydrogens_and_non_atom_records(tmp_path: Path) -> None:
    """Max-ASA reference values are heavy-atom surfaces, so hydrogens must not count."""
    path = tmp_path / "model.pdb"
    path.write_text(
        "HEADER    TEST\n"
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 88.00           N\n"
        "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00 90.00           C\n"
        "ATOM      3  H   ALA A   1      12.000   6.000  -5.000  1.00 90.00           H\n"
        "HETATM    4  O   HOH B   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    atoms = load_atoms(path)
    assert [atom.name for atom in atoms] == ["N", "CA"]
    assert atoms[1].plddt == pytest.approx(90.0)


def test_empty_structure_is_not_an_error() -> None:
    """No atoms means no surface and no pockets, not a crash."""
    assert atom_sasa([]).size == 0
    assert residue_surfaces([], np.zeros(0)) == []
    assert find_pockets([], []) == []
