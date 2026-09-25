"""The guarantee that candidate models cannot affect production results.

`experimental/` holds challengers under evaluation. The value of a head-to-head bake-off
depends entirely on the incumbent being unaffected by the challenger's presence - if
importing a candidate could change a production number, the comparison would be measuring
the two models plus their interaction.

These tests enforce that structurally rather than trusting it, because the failure mode is
silent: an import added during a refactor would not break anything visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experimental.grammar_classifier import grammar_features

REPO_ROOT = Path(__file__).resolve().parents[1]

# Packages that must never depend on a candidate model.
#
# `validation` is deliberately absent: the bake-off is what compares a challenger against
# the incumbent, so it has to import both. What matters is that the *incumbent* cannot see
# the challenger, which is exactly what the packages below assert. If `domain_layout` ever
# imported a candidate, the comparison would be measuring the two models plus whatever
# they did to each other.
PRODUCTION_PACKAGES = ("domain_layout", "jdp_classifier", "motif_conservation", "data_fetching", "structure_analysis")


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", PRODUCTION_PACKAGES)
def test_production_code_never_imports_a_candidate_model(package: str) -> None:
    """The one-way dependency that makes the bake-off meaningful."""
    for source in sorted((REPO_ROOT / package).rglob("*.py")):
        for imported in _imported_modules(source):
            assert not imported.startswith("experimental"), (
                f"{source.relative_to(REPO_ROOT)} imports {imported}; production code must not "
                f"depend on a candidate model."
            )


def test_the_challenger_uses_no_domain_annotation() -> None:
    """The challenger's claim is that it works without annotation, so verify it.

    If it read InterPro entries, its score would not represent what is available for an
    unannotated protein, and the comparison it exists to make would be meaningless.
    """
    source = (REPO_ROOT / "experimental" / "grammar_classifier.py").read_text(encoding="utf-8")
    for forbidden in (".entries", "domain_family_layout", "has_dnaj_c", "has_zinc_finger", "n_structured_domains"):
        assert forbidden not in source, f"challenger reads {forbidden}, which is annotation-derived"


def test_challenger_features_are_computable_without_any_annotation() -> None:
    """Exercise the claim rather than only grepping for it."""
    features = grammar_features("MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG")
    assert features
    assert all(isinstance(value, float) for value in features.values())


def test_experimental_package_documents_its_status() -> None:
    """Anyone opening the package must learn immediately that it is not production."""
    doc = (REPO_ROOT / "experimental" / "__init__.py").read_text(encoding="utf-8")
    assert "challenger" in doc.lower()
    assert "production" in doc.lower()


def test_validation_may_import_a_candidate_but_domain_layout_may_not() -> None:
    """The asymmetry that makes the bake-off meaningful, pinned so it stays deliberate."""
    validation_imports = {
        name for source in sorted((REPO_ROOT / "validation").rglob("*.py")) for name in _imported_modules(source)
    }
    # The comparison lives in validation, so it necessarily reaches the challenger.
    assert any(name.startswith("experimental") for name in validation_imports)

    layout_imports = {
        name for source in sorted((REPO_ROOT / "domain_layout").rglob("*.py")) for name in _imported_modules(source)
    }
    assert not any(name.startswith("experimental") for name in layout_imports)
    assert not any(name.startswith("validation") for name in layout_imports)


def test_the_challenger_carries_no_absolute_length_feature() -> None:
    """Class C is largely "J-domain only", i.e. shorter, so raw length is a leak.

    A model that learns length would score well on this label set while learning nothing
    about grammar, and would not transfer to an organism whose JDPs differ in size.
    """
    short = grammar_features("MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN" * 2)
    long_one = grammar_features("MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN" * 20)

    for name in short:
        assert "length" not in name, f"feature '{name}' names length directly"

    # No feature may scale with length the way a log-length term would.
    for name, value in short.items():
        other = long_one[name]
        assert abs(other - value) < 5.0, f"feature '{name}' moved {value} -> {other} on a 10x length change"
