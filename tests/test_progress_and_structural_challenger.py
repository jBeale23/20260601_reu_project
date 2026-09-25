"""Tests for progress reporting and the two structural challenger variants."""

from __future__ import annotations

import sys

import pytest

from domain_layout.progress import progress
from experimental.structural_classifier import (
    EXCLUDE_MISSING,
    IMPUTE_MISSING,
    STRUCTURAL_FEATURES,
    agreement_note,
    combined_features,
    load_structural_features,
    variants,
)


class _Record:
    def __init__(self, accession: str, sequence: str) -> None:
        self.accession = accession
        self.sequence = sequence


class _Layout:
    def __init__(self, accession: str) -> None:
        self.record = _Record(accession, "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN" * 3)
        self.regions = ()


def test_progress_passes_every_item_through() -> None:
    """A progress wrapper that changed the iteration would be a bug, not a display."""
    assert list(progress(range(5), description="test")) == [0, 1, 2, 3, 4]
    assert list(progress([], description="empty")) == []


def test_progress_is_silent_when_stderr_is_not_a_terminal(capsys: pytest.CaptureFixture[str]) -> None:
    """A bar redrawing thousands of times fills a cluster log with control characters."""
    assert not sys.stderr.isatty()
    list(progress(range(50), description="quiet"))
    assert capsys.readouterr().err == ""


def test_structural_table_is_read_and_length_normalised(tmp_path) -> None:  # noqa: ANN001
    """Total surface area scales with size, so it cannot enter the model unnormalised."""
    path = tmp_path / "s.csv"
    path.write_text(
        "accession,n_residues,mean_plddt,fraction_buried,total_sasa,largest_pocket_volume\n"
        "P1,100,80.0,0.30,10000,400\n"
        "P2,200,80.0,0.30,20000,800\n",
        encoding="utf-8",
    )
    features = load_structural_features(path)

    assert set(features) == {"P1", "P2"}
    # Twice the size and twice the area gives the same per-residue value.
    assert features["P1"]["total_sasa_per_residue"] == pytest.approx(features["P2"]["total_sasa_per_residue"])
    assert features["P1"]["mean_plddt"] == pytest.approx(80.0)


def test_excluding_variant_drops_proteins_without_a_model() -> None:
    """Honest about what was measured, at the cost of a smaller and more biased set."""
    layouts = [_Layout("P1"), _Layout("P2"), _Layout("NO_MODEL")]
    structural = {"P1": {"mean_plddt": 80.0}, "P2": {"mean_plddt": 70.0}}

    features, coverage = combined_features(layouts, structural, impute_missing=False)

    assert set(features) == {"P1", "P2"}
    assert coverage.n_proteins == 2
    assert coverage.coverage == pytest.approx(1.0)


def test_imputing_variant_keeps_every_protein_and_flags_the_invented_rows() -> None:
    """Imputation without a flag would let an invented value pass as a measurement."""
    layouts = [_Layout("P1"), _Layout("P2"), _Layout("NO_MODEL")]
    structural = {"P1": {"mean_plddt": 90.0}, "P2": {"mean_plddt": 70.0}}

    features, coverage = combined_features(layouts, structural, impute_missing=True)

    assert set(features) == {"P1", "P2", "NO_MODEL"}
    assert features["NO_MODEL"]["has_structure"] == 0.0
    assert features["P1"]["has_structure"] == 1.0
    # The filled value is the column median, not zero, which would be a real pLDDT.
    assert features["NO_MODEL"]["mean_plddt"] == pytest.approx(80.0)
    assert coverage.n_proteins == 3
    assert coverage.n_with_structure == 2


def test_both_variants_are_produced_together() -> None:
    """Running only one leaves the effect of missing models unmeasurable."""
    layouts = [_Layout("P1"), _Layout("NO_MODEL")]
    built = variants(layouts, {"P1": {"mean_plddt": 80.0}})

    assert set(built) == {EXCLUDE_MISSING, IMPUTE_MISSING}
    assert built[EXCLUDE_MISSING][1].n_proteins == 1
    assert built[IMPUTE_MISSING][1].n_proteins == 2


def test_grammar_features_are_kept_alongside_the_structural_ones() -> None:
    """The structural challenger extends the grammar model rather than replacing it."""
    features, _coverage = combined_features([_Layout("P1")], {"P1": {"mean_plddt": 80.0}}, impute_missing=False)
    row = features["P1"]
    assert "mean_plddt" in row
    # A grammar feature must still be present.
    assert any(name.startswith("chemical7") for name in row)


def test_agreement_note_says_what_a_difference_means() -> None:
    """Two numbers without an interpretation invite the reader to pick the nicer one."""
    close = agreement_note(0.900, 0.905)
    apart = agreement_note(0.900, 0.960)
    assert "agree" in close
    assert "either number can be quoted" in close
    assert "model availability" in apart
    assert "neither number should be quoted without the other" in apart


def test_structural_feature_list_carries_no_absolute_length() -> None:
    """Raw length was a leak in the grammar model; it must not return through structure."""
    for name in STRUCTURAL_FEATURES:
        assert "n_residues" not in name
        assert name not in {"total_sasa", "largest_pocket_volume"}
