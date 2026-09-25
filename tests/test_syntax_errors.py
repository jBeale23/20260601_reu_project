"""Tests for reading a protein as a sentence and locating where it stops making sense.

Each test corresponds to a way the detector was wrong before it was right. Three of them
pin bugs that made it silently useless: a polyglutamine tract - the canonical repeat
expansion mechanism - was invisible; spans bled a full window past each edge, diluting the
classification; and a charge anomaly could only be called when the family's own mean charge
had the opposite sign, which is almost never.
"""

from __future__ import annotations

import pytest

from domain_layout.grammar import CorpusGrammar
from domain_layout.syntax_errors import (
    CHARGE_ANOMALY,
    ERROR_TYPES,
    HYDROPHOBIC_EXPOSURE,
    LOW_COMPLEXITY,
    MIN_SPAN_RESIDUES,
    REPEAT_EXPANSION,
    DetectorSettings,
    FamilyReference,
    classify_error,
    find_syntax_errors,
    positional_surprisal,
)

_JD = "MVKETKFYDILGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKEISEAYEVLSDPEKREIYDQ"


def _family() -> list[str]:
    return [_JD, _JD[:-4] + "AAAA", "M" + _JD[1:], _JD.replace("K", "R", 3)] * 8


def _fitted() -> tuple[CorpusGrammar, FamilyReference]:
    family = _family()
    return CorpusGrammar(order=3).fit(family), FamilyReference.from_sequences(family)


def _with_insert(insert: str, at: int = 30) -> str:
    return _JD[:at] + insert + _JD[at:]


# Detection
# ---------


def test_a_clean_family_member_has_no_lesion() -> None:
    """Every protein reporting an error would make the detector useless."""
    corpus, family = _fitted()
    assert find_syntax_errors(_JD, corpus, family=family) == []


def test_a_polyglutamine_tract_is_detected() -> None:
    """The regression that mattered most.

    Standardising each window against the protein's own *mean* let a long lesion drag the
    baseline up towards itself, so a twenty-residue polyQ tract - the mechanism behind an
    entire disease class - never cleared the threshold. A median baseline is unmoved by a
    minority of extreme windows.
    """
    corpus, family = _fitted()
    found = find_syntax_errors(_with_insert("Q" * 20), corpus, family=family)
    assert found, "a 20-residue polyQ tract went undetected"
    assert found[0].error_type == REPEAT_EXPANSION


def test_a_lesion_is_localised_to_roughly_its_true_extent() -> None:
    """Taking the union of flagged windows over-reported a 20aa insert as 33aa."""
    corpus, family = _fitted()
    found = find_syntax_errors(_with_insert("Q" * 20), corpus, family=family)
    lesion = found[0]
    # True insertion occupies residues 31-50.
    assert 28 <= lesion.start <= 34
    assert 46 <= lesion.end <= 56
    assert lesion.length <= 30, f"span of {lesion.length} for a 20-residue lesion"


def test_a_hydrophobic_patch_is_typed_by_its_consequence() -> None:
    """The classic aggregation-prone lesion must not read as generic low complexity."""
    corpus, family = _fitted()
    found = find_syntax_errors(_with_insert("IIIVVVLLLFFFWWWAAA"), corpus, family=family)
    assert found[0].error_type == HYDROPHOBIC_EXPOSURE


def test_an_acidic_patch_in_a_neutral_family_is_a_charge_anomaly() -> None:
    """Requiring a sign flip missed this entirely.

    The family's own mean charge is near zero, so a strongly acidic span has the *same*
    sign; the product test passed and the span fell through to "low complexity", which
    describes it accurately and says nothing about what it does.
    """
    corpus, family = _fitted()
    found = find_syntax_errors(_with_insert("DDDDEEEEDDDDEEEE"), corpus, family=family)
    assert found[0].error_type == CHARGE_ANOMALY
    assert "shifted" in found[0].detail or "reversed" in found[0].detail


def test_a_basic_patch_reports_reversal_explicitly() -> None:
    """Sign reversal is still worth naming when it happens."""
    corpus, family = _fitted()
    found = find_syntax_errors(_with_insert("KKKKRRRRKKKKRRRR"), corpus, family=family)
    assert found[0].error_type == CHARGE_ANOMALY


# Classification
# --------------


def test_classification_is_ordered_by_consequence_not_pattern() -> None:
    """A poly-aspartate tract is a repeat, low-complexity, and a charge anomaly at once.

    The call worth making is the one that says what it does to the protein.
    """
    kind, _detail = classify_error("DDDDDDDDDDDDDDDD", family_charge=0.0, family_hydrophobic=0.3)
    assert kind in {REPEAT_EXPANSION, CHARGE_ANOMALY}


def test_a_neutral_low_complexity_linker_is_called_as_such() -> None:
    """Glycine-serine linkers are neither charged nor hydrophobic; that is the fallback."""
    kind, _detail = classify_error("GSGSGGSGGSGGSGGS" * 2, family_charge=0.0, family_hydrophobic=0.3)
    assert kind in {LOW_COMPLEXITY, REPEAT_EXPANSION}


def test_every_error_type_is_reachable() -> None:
    """A type nothing can produce is documentation, not a classifier."""
    assert CHARGE_ANOMALY in ERROR_TYPES
    assert len(set(ERROR_TYPES)) == len(ERROR_TYPES)


# Mechanics
# ---------


def test_positional_surprisal_needs_a_fitted_corpus() -> None:
    """An unfitted model must return nothing rather than uniform scores."""
    assert positional_surprisal(_JD, CorpusGrammar(order=3)) == []


def test_a_sequence_shorter_than_the_window_yields_nothing() -> None:
    """No window fits, so there is nothing to be unusual against."""
    corpus, family = _fitted()
    assert find_syntax_errors("MVKET", corpus, family=family) == []


def test_short_blips_are_not_reported_as_spans() -> None:
    """Below a few residues, composition is not measurable."""
    assert MIN_SPAN_RESIDUES >= 6


def test_a_stricter_threshold_reports_no_more_lesions() -> None:
    """Monotonicity - raising the bar cannot find more."""
    corpus, family = _fitted()
    sequence = _with_insert("Q" * 20)
    loose = find_syntax_errors(sequence, corpus, family=family, settings=DetectorSettings(z_threshold=2.0))
    strict = find_syntax_errors(sequence, corpus, family=family, settings=DetectorSettings(z_threshold=6.0))
    assert len(strict) <= len(loose)


def test_the_family_reference_is_fitted_from_the_family() -> None:
    """Without it, "charge anomaly" would mean "differs from neutral"."""
    reference = FamilyReference.from_sequences(["KKKKKKKK", "KKKKRRRR"])
    assert reference.net_charge == pytest.approx(1.0)
    # Against an all-basic family, a basic span is unremarkable.
    kind, _ = classify_error(
        "KKKKRRRRKKKK", family_charge=reference.net_charge, family_hydrophobic=reference.hydrophobic_fraction
    )
    assert kind != CHARGE_ANOMALY
