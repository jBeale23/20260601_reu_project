"""Tests for measuring grammar separately in disordered and folded regions.

Roughly half of J-domain protein sequence is unalignable, so a feature averaged over the
whole chain describes two kinds of sequence at once and belongs to neither. These tests pin
the separation and, in particular, pin that a comparison is refused when only one side of
it exists.
"""

from __future__ import annotations

import pytest

from domain_layout.regions import ROUTE_MSA, ROUTE_SHARK
from validation.grammar_compartments import (
    COMPARTMENTS,
    DISORDERED,
    MIN_COMPARTMENT_LENGTH,
    STRUCTURED,
    WHOLE,
    compartment_divergence,
    compartment_report,
    compartment_sequences,
    features_for_correlation,
    grammar_by_compartment,
)

_FOLDED = "MVKETKFYDILGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKEISEAYEVLSDPEKREIYDQ"
_LINKER = "GGSGSGGSGGSGSGGSGGSGSGGSGGSGSGGSGGSGSGGSGGSGSGGSGGSGSGGSGGSGSGGSGG"


class _Region:
    def __init__(self, sequence: str, route: str) -> None:
        self.sequence = sequence
        self.route = route


class _Record:
    def __init__(self, accession: str, sequence: str) -> None:
        self.accession = accession
        self.sequence = sequence


class _Layout:
    def __init__(self, accession: str, regions: list[_Region]) -> None:
        self.regions = regions
        self.record = _Record(accession, "".join(r.sequence for r in regions))


def _layout(accession: str = "P1") -> _Layout:
    return _Layout(
        accession,
        [_Region(_FOLDED, ROUTE_MSA), _Region(_LINKER, ROUTE_SHARK), _Region(_FOLDED, ROUTE_MSA)],
    )


def test_the_two_compartments_are_separated_by_route() -> None:
    """Disordered and folded sequence must not end up in the same measurement."""
    disordered, structured = compartment_sequences(_layout())
    assert disordered == _LINKER
    assert structured == _FOLDED * 2


def test_features_are_computed_on_concatenated_sequence() -> None:
    """One 300-residue linker is a different quantity from six 50-residue pieces.

    The former is what "how is this protein's disordered sequence written" means.
    """
    result = grammar_by_compartment(_layout())
    assert result is not None
    assert result.n_disordered_residues == len(_LINKER)
    assert result.n_structured_residues == 2 * len(_FOLDED)


def test_a_linker_and_a_domain_are_grammatically_distinguishable() -> None:
    """The sanity check before anything downstream is claimed.

    If the two compartments were indistinguishable, the routing that produced them would
    not be separating anything and every later comparison would be measuring noise.
    """
    result = grammar_by_compartment(_layout())
    assert result is not None
    # A glycine/serine linker is far less compositionally diverse than a folded domain.
    shared = set(result.disordered) & set(result.structured)
    differing = [name for name in shared if abs(result.disordered[name] - result.structured[name]) > 0.05]
    assert differing, "no feature separated a GS linker from a J-domain"


def test_a_protein_missing_a_compartment_is_refused() -> None:
    """A comparison needs both sides.

    Fully disordered and fully folded proteins are real and interesting, but they cannot
    answer a question about the difference between the two, and silently returning
    half a comparison would put them in the denominator anyway.
    """
    folded_only = _Layout("P2", [_Region(_FOLDED, ROUTE_MSA)])
    assert grammar_by_compartment(folded_only) is None


def test_a_compartment_below_the_length_floor_is_refused() -> None:
    """Below a few dozen residues the entropy spectrum measures length, not arrangement."""
    tiny = _Layout("P3", [_Region(_FOLDED, ROUTE_MSA), _Region("GGSGG", ROUTE_SHARK)])
    assert grammar_by_compartment(tiny) is None
    assert MIN_COMPARTMENT_LENGTH >= 40


def test_divergence_reports_a_median_gap_per_feature() -> None:
    """Median rather than mean, because a handful of odd proteins should not set it."""
    grammars = [g for g in (grammar_by_compartment(_layout(f"P{i}")) for i in range(5)) if g]
    divergence = compartment_divergence(grammars)
    assert divergence
    assert all(isinstance(value, float) for value in divergence.values())


def test_every_compartment_is_addressable() -> None:
    """Whole-sequence features are kept too, so the pooled measure can be compared against."""
    result = grammar_by_compartment(_layout())
    assert result is not None
    assert set(COMPARTMENTS) == {DISORDERED, STRUCTURED, WHOLE}
    for compartment in COMPARTMENTS:
        assert result.by_compartment(compartment)


def test_an_unknown_compartment_is_refused() -> None:
    """Silently returning nothing would look like a protein set with no features."""
    grammars = [g for g in (grammar_by_compartment(_layout()),) if g]
    with pytest.raises(ValueError, match="unknown compartment"):
        features_for_correlation(grammars, "somewhere_else")


def test_the_report_states_who_is_excluded() -> None:
    """Fully disordered and fully folded proteins are absent and that must be visible."""
    grammars = [g for g in (grammar_by_compartment(_layout(f"P{i}")) for i in range(3)) if g]
    report = compartment_report(grammars)
    assert report["n_proteins"] == 3
    assert "fully disordered and fully folded proteins are absent" in str(report["note"])


def test_an_empty_set_reports_rather_than_crashes() -> None:
    """No protein having both compartments is a result, not an error."""
    assert compartment_report([])["n_proteins"] == 0
