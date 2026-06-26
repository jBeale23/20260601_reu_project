"""Regression: Rockfish SLURM scripts preserve critical safety patterns."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts" / "slurm"


def _read_slurm(name: str) -> str:
    return (SLURM_DIR / name).read_text(encoding="utf-8")


def test_submit_affetch_cds_before_python_queue() -> None:
    """Queue helpers must run from PROJECT_DIR (Bugbot: cwd before python -m)."""
    text = _read_slurm("submit_affetch_rockfish.sh")
    cd_idx = text.index('cd "${PROJECT_DIR}"')
    python_idx = text.index("python -m scripts.rockfish_queue")
    assert cd_idx < python_idx


def test_submit_analyze_pocket_cds_before_python_queue() -> None:
    """Pocket submit script runs queue helpers from PROJECT_DIR."""
    text = _read_slurm("submit_analyze_pocket_rockfish.sh")
    cd_idx = text.index('cd "${PROJECT_DIR}"')
    python_idx = text.index("python -m scripts.rockfish_queue")
    assert cd_idx < python_idx


def test_affetch_worker_exits_nonzero_on_download_failure() -> None:
    """Failed affetch downloads must fail the SLURM task (not exit 0)."""
    text = _read_slurm("affetch_rockfish.sh")
    failure_block = text.split("if affetch", maxsplit=1)[1]
    assert "exit 1" in failure_block


def test_analyze_pocket_worker_exits_nonzero_on_analysis_failure() -> None:
    """Failed pocket analysis must fail the SLURM task."""
    text = _read_slurm("analyze_pocket_rockfish.sh")
    failure_block = text.split("if analyze-pocket-charge", maxsplit=1)[1]
    assert "exit 1" in failure_block


def test_rockfish_common_uses_flock_for_completion_log() -> None:
    """Completion and failure logging must be atomic to avoid duplicate lines."""
    text = _read_slurm("rockfish_common.sh")
    assert "flock" in text
    assert "rockfish_mark_completed" in text
    assert "rockfish_log_failure" in text
    assert "rockfish_resolve_structure" in text
    assert "rockfish_resolve_pdb_structure" in text


@pytest.mark.parametrize(
    "script_name",
    ["submit_affetch_rockfish.sh", "submit_analyze_pocket_rockfish.sh"],
)
def test_submit_scripts_use_array_queue_snapshot(script_name: str) -> None:
    """Submit launchers pass a fixed snapshot, not a live pending list."""
    text = _read_slurm(script_name)
    assert "write-snapshot" in text
    assert "ARRAY_QUEUE_FILE" in text
