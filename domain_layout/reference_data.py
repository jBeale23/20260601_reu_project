"""Locate the bundled curated reference JDP domain store."""

from __future__ import annotations

from pathlib import Path

REFERENCE_DIRNAME = Path("data") / "reference_jdps"
REFERENCE_STORE_FILENAME = "reference_domain_store.json"
REFERENCE_CLASSES_FILENAME = "reference_classes.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def default_reference_dir() -> Path:
    """Return the bundled reference directory (may not exist in a wheel install)."""
    return _REPO_ROOT / REFERENCE_DIRNAME


def default_reference_store() -> Path:
    """Path to the curated reference domain store."""
    return default_reference_dir() / REFERENCE_STORE_FILENAME


def default_reference_classes() -> Path:
    """Path to the curated reference class map."""
    return default_reference_dir() / REFERENCE_CLASSES_FILENAME


def reference_data_available() -> bool:
    """Whether both bundled reference files are present."""
    return default_reference_store().is_file() and default_reference_classes().is_file()
