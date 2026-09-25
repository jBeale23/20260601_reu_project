"""Regression: every pyproject console script target is importable and callable."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from validation.cli import load_structural_features

REPO_ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from collections.abc import Callable

# Mirrors [project.scripts] in pyproject.toml — update when adding CLIs.
CONSOLE_SCRIPT_TARGETS: dict[str, str] = {
    "fetch-proteins-dnak": "data_fetching.fetch_proteins_dnak:cli",
    "fetch-architectures-dnaj": "data_fetching.fetch_architectures_dnaj:cli",
    "fetch-protein-domains": "data_fetching.fetch_domains:cli",
    "analyze-domain-layout": "domain_layout.cli:main",
    "validate-jdp-classification": "validation.cli:main",
    "extract-uniprot-ids": "scripts.extract_uniprot_ids:main",
    "prepare-rockfish-accessions": "scripts.rockfish_queue:main_prepare",
    "summarize-rockfish-failures": "scripts.rockfish_queue:main_summarize_failures",
    "merge-features": "scripts.merge_features:main",
    "merge-all-features": "scripts.merge_all_features:main",
    "classify-jdp": "jdp_classifier.cli:main",
    "analyze-pocket-charge": "structure_analysis.cli:main",
    "analyze-protein-structures": "structure_analysis.features_cli:main",
    "analyze-motif-conservation": "motif_conservation.cli:main",
}


def _load_callable(target: str) -> Callable[[], object]:
    module_name, attr_name = target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


@pytest.mark.parametrize(("script_name", "target"), CONSOLE_SCRIPT_TARGETS.items())
def test_console_script_entry_point_is_callable(script_name: str, target: str) -> None:
    """Each declared CLI entry point resolves to a callable function."""
    fn = _load_callable(target)
    assert callable(fn), f"{script_name} -> {target} is not callable"


def test_every_top_level_package_is_declared_for_packaging() -> None:
    """A package missing from pyproject imports locally and fails once installed.

    Local runs put the repository root on sys.path, so an undeclared package works right
    up until it is installed with `pip install -e .` and run somewhere else - which is how
    `experimental` reached a cluster job before this test existed.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    declared_names = {entry.rstrip("*") for entry in declared}

    for candidate in sorted(REPO_ROOT.iterdir()):
        if not candidate.is_dir() or not (candidate / "__init__.py").is_file():
            continue
        if candidate.name in {"tests", "docs", "vendor"} or candidate.name.startswith("."):
            continue
        assert candidate.name in declared_names, (
            f"package '{candidate.name}' has an __init__.py but is not in "
            f"[tool.setuptools.packages.find].include; it will be missing once installed."
        )


def test_structural_features_load_from_the_csv_the_structure_stage_writes(tmp_path: Path) -> None:
    """The bake-off's structural challengers are enabled by this option and nothing else.

    They had never run: the features were computed and written to CSV, but no command-line
    option handed them to the report, so ``structural_features`` was always empty and the
    two structural variants were silently skipped.
    """
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "accession,mean_plddt,fraction_buried,is_largely_disordered\nP0A6Y8,88.5,0.42,False\nQ9UBS3,45.1,0.11,True\n",
        encoding="utf-8",
    )
    features = load_structural_features(csv_path)
    assert set(features) == {"P0A6Y8", "Q9UBS3"}
    assert features["P0A6Y8"]["mean_plddt"] == 88.5
    # A boolean flag must not be coerced into a silent zero.
    assert "is_largely_disordered" not in features["P0A6Y8"]


def test_no_structural_csv_means_a_grammar_only_bakeoff() -> None:
    """The default has to stay empty, or a run without structures would claim to have them."""
    assert load_structural_features(None) == {}


def test_rows_without_an_accession_are_skipped(tmp_path: Path) -> None:
    """A blank key would collect features under an empty accession and match nothing."""
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("accession,mean_plddt\n,50.0\nP1,70.0\n", encoding="utf-8")
    assert set(load_structural_features(csv_path)) == {"P1"}
