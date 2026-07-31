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
    assert "affetch_nonzero" in failure_block


def test_analyze_pocket_worker_exits_nonzero_on_analysis_failure() -> None:
    """Failed pocket analysis must fail the SLURM task."""
    text = _read_slurm("analyze_pocket_rockfish.sh")
    failure_block = text.split("if analyze-pocket-charge", maxsplit=1)[1]
    assert "exit 1" in failure_block
    assert "analyze_nonzero" in failure_block


def test_analyze_pocket_worker_diagnoses_missing_pdb() -> None:
    """Missing structure path must log a descriptive reason code."""
    text = _read_slurm("analyze_pocket_rockfish.sh")
    assert "rockfish_diagnose_missing_pdb" in text
    assert "cif_only" in _read_slurm("rockfish_common.sh")


def test_rockfish_common_uses_flock_for_completion_log() -> None:
    """Completion and failure logging must be atomic to avoid duplicate lines."""
    text = _read_slurm("rockfish_common.sh")
    assert "flock" in text
    assert "rockfish_mark_completed" in text
    assert "rockfish_log_failure" in text
    assert "rockfish_resolve_structure" in text
    assert "rockfish_resolve_pdb_structure" in text
    assert "rockfish_diagnose_missing_pdb" in text
    assert "reason_code" in text or "reason_code" in text.lower() or "TAB" in text or "\\t" in text


def test_submit_scripts_exclude_failed_via_python_snapshot() -> None:
    """Submit launchers pass --failed into Python write-snapshot (not bash-only comm)."""
    for script_name in ("submit_affetch_rockfish.sh", "submit_analyze_pocket_rockfish.sh"):
        text = _read_slurm(script_name)
        assert "write-snapshot" in text
        assert "--failed" in text
        assert "ARRAY_QUEUE_FILE" in text
        assert "RETRY_FAILED" in text


def test_submit_pocket_uses_pdb_preflight() -> None:
    """Pocket submit filters to PDB-backed accessions by default."""
    text = _read_slurm("submit_analyze_pocket_rockfish.sh")
    assert "--require-pdb" in text
    assert "REQUIRE_PDB" in text
    assert "skipped_no_pdb" in text


@pytest.mark.parametrize(
    "script_name",
    ["submit_affetch_rockfish.sh", "submit_analyze_pocket_rockfish.sh"],
)
def test_submit_scripts_use_array_queue_snapshot(script_name: str) -> None:
    """Submit launchers pass a fixed snapshot, not a live pending list."""
    text = _read_slurm(script_name)
    assert "write-snapshot" in text
    assert "ARRAY_QUEUE_FILE" in text


def test_submit_motif_script_exists_and_exports_fetch_json() -> None:
    """Motif Rockfish launcher points at analyze_motif_rockfish.sh."""
    text = _read_slurm("submit_analyze_motif_rockfish.sh")
    assert "analyze_motif_rockfish.sh" in text
    assert "FETCH_JSON" in text
    assert "motif_results" in text


def test_analyze_motif_worker_uses_reason_coded_failure() -> None:
    """Motif worker logs analyze_nonzero on failure."""
    text = _read_slurm("analyze_motif_rockfish.sh")
    assert "analyze-motif-conservation" in text
    assert "analyze_nonzero" in text
    assert "rockfish_log_failure" in text
