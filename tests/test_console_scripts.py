"""Regression: every pyproject console script target is importable and callable."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Mirrors [project.scripts] in pyproject.toml — update when adding CLIs.
CONSOLE_SCRIPT_TARGETS: dict[str, str] = {
    "fetch-proteins-dnak": "data_fetching.fetch_proteins_dnak:cli",
    "fetch-architectures-dnaj": "data_fetching.fetch_architectures_dnaj:cli",
    "extract-uniprot-ids": "scripts.extract_uniprot_ids:main",
    "prepare-rockfish-accessions": "scripts.rockfish_queue:main_prepare",
    "merge-features": "scripts.merge_features:main",
    "merge-all-features": "scripts.merge_all_features:main",
    "classify-jdp": "jdp_classifier.cli:main",
    "analyze-pocket-charge": "structure_analysis.cli:main",
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
