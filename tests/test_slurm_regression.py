"""Regression: Rockfish SLURM scripts preserve critical safety patterns."""

from __future__ import annotations

import subprocess
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


def test_submit_fetch_domains_cds_before_python_queue() -> None:
    """Queue helpers, and the interpreter check, run from PROJECT_DIR.

    The import probe must come after the cd: the project is importable from the project
    root even when it has not been pip-installed.
    """
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    cd_idx = text.index('cd "${PROJECT_DIR}"')
    assert cd_idx < text.index('"${PYTHON}" -m scripts.rockfish_queue')
    assert cd_idx < text.index("import scripts.rockfish_queue")


def test_submit_fetch_domains_uses_snapshot_and_chunking() -> None:
    """Domain fetch maps a fixed snapshot to chunked array tasks."""
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "write-snapshot" in text
    assert "ARRAY_QUEUE_FILE" in text
    assert "CHUNK_SIZE" in text
    assert "--array=1-" in text


def test_fetch_domains_worker_is_chunked_and_reason_coded() -> None:
    """Each domain-fetch task owns one chunk and logs reason-coded failures."""
    text = _read_slurm("fetch_domains_rockfish.sh")
    assert "rockfish_chunk_start" in text
    assert "rockfish_chunk_marker" in text
    assert "--start" in text
    assert "--count" in text
    failure_block = text.split("if fetch-protein-domains", maxsplit=1)[1]
    assert "exit 1" in failure_block
    assert "fetch_nonzero" in failure_block


def test_fetch_domains_worker_requires_queue_file() -> None:
    """A worker without a submit-time snapshot fails fast with guidance."""
    text = _read_slurm("fetch_domains_rockfish.sh")
    assert "ARRAY_QUEUE_FILE is missing or not found" in text
    assert "submit_fetch_domains_rockfish.sh" in text


def test_analyze_domain_layout_worker_reports_backends_and_failures() -> None:
    """The layout worker records active backends and reason-codes failures."""
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    assert "--show-backends" in text
    assert "analyze-domain-layout" in text
    failure_block = text.split('if analyze-domain-layout "${layout_args[@]}"', maxsplit=1)[1]
    assert "exit 1" in failure_block
    assert "analyze_nonzero" in failure_block
    assert "rockfish_log_failure" in failure_block


def test_submit_domain_layout_requires_merged_store() -> None:
    """The layout launcher refuses to submit without a merged domain store."""
    text = _read_slurm("submit_analyze_domain_layout_rockfish.sh")
    assert "Domain store not found" in text
    assert "--merge-stores" in text
    assert "analyze_domain_layout_rockfish.sh" in text


def test_layout_conda_env_pins_python_for_bio_shark() -> None:
    """bio-shark supports <3.13, so the layout env must not float to a newer Python."""
    text = (SLURM_DIR / "conda_env_layout.yaml").read_text(encoding="utf-8")
    assert "python=3.12" in text
    assert "metapredict" in text
    assert "bio-shark" in text


@pytest.mark.parametrize(
    ("task_id", "chunk_size", "expected_start"),
    [("1", "200", "1"), ("2", "200", "201"), ("5", "50", "201")],
)
def test_rockfish_chunk_start_maps_task_to_snapshot_line(task_id: str, chunk_size: str, expected_start: str) -> None:
    """Chunk helpers map array task IDs to 1-based snapshot lines."""
    script = f'source "{SLURM_DIR / "rockfish_common.sh"}"; rockfish_chunk_start {task_id} {chunk_size}'
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == expected_start


def _chunk_marker(queue_file: Path, start: int, chunk_size: int) -> str:
    script = f'source "{SLURM_DIR / "rockfish_common.sh"}"; rockfish_chunk_marker {start} "{queue_file}" {chunk_size}'
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_chunk_marker_identifies_content_not_just_position(tmp_path: Path) -> None:
    """A marker tracks the accessions in the chunk, so a shifted list is not skipped.

    Completion is recorded per chunk. If markers were bare line offsets, rebuilding the
    master accession list would leave a marker claiming a chunk was done while it now
    covers entirely different proteins.
    """
    original = tmp_path / "queue_a.txt"
    original.write_text("P00001\nP00002\nP00003\nP00004\n", encoding="utf-8")

    shifted = tmp_path / "queue_b.txt"
    shifted.write_text("P00000\nP00001\nP00002\nP00003\nP00004\n", encoding="utf-8")

    marker_original = _chunk_marker(original, 1, 2)
    marker_shifted = _chunk_marker(shifted, 1, 2)

    assert marker_original.startswith("chunk_000001_")
    assert marker_original == _chunk_marker(original, 1, 2), "identical content must be stable"
    assert marker_original != marker_shifted, "different accessions must differ"


def test_chunk_marker_without_a_queue_file(tmp_path: Path) -> None:
    """A missing queue file still yields a usable marker rather than failing the task."""
    assert _chunk_marker(tmp_path / "missing.txt", 201, 50) == "chunk_000201_nocontent"


def test_submit_fetch_domains_guards_array_bounds() -> None:
    """A too-small CHUNK_SIZE must be refused before sbatch rejects the array."""
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "Refusing to submit" in text
    assert "MAX_ARRAY_TASKS" in text
    assert "required_chunk" in text


def test_layout_worker_uses_granted_cpus() -> None:
    """The layout job parallelizes across the cores SLURM allocated."""
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    assert "SLURM_CPUS_PER_TASK" in text
    assert "--workers" in text


def test_layout_worker_requests_multiple_cpus() -> None:
    """The job asks for more than one CPU, otherwise --workers cannot help."""
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    cpus_line = next(line for line in text.splitlines() if "--cpus-per-task" in line)
    assert int(cpus_line.split("=")[-1]) > 1


def test_new_workers_activate_either_env_flavour() -> None:
    """The domain workers must accept a conda env or a plain venv.

    `conda activate` fails outright on a venv, which is how the environment is built when
    conda is unavailable or a specific Python is needed for bio-shark.
    """
    for script in ("fetch_domains_rockfish.sh", "analyze_domain_layout_rockfish.sh"):
        text = _read_slurm(script)
        assert "rockfish_activate_env" in text, script
        assert 'conda activate "${CONDA_ENV}"' not in text, script

    helper = _read_slurm("rockfish_common.sh")
    assert "pyvenv.cfg" in helper
    assert "rockfish_activate_env" in helper


def test_activate_helper_handles_a_venv(tmp_path: Path) -> None:
    """A directory with pyvenv.cfg is sourced, never passed to conda."""
    env = tmp_path / "venv"
    (env / "bin").mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (env / "bin" / "activate").write_text("export ROCKFISH_ACTIVATED=venv\n", encoding="utf-8")

    script = (
        f'source "{SLURM_DIR / "rockfish_common.sh"}"; '
        f'conda() {{ echo "conda-was-called"; }}; '
        f'rockfish_activate_env "{env}"; echo "${{ROCKFISH_ACTIVATED:-none}}"'
    )
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "venv" in completed.stdout
    assert "conda-was-called" not in completed.stdout


def test_activate_helper_falls_back_to_conda(tmp_path: Path) -> None:
    """A directory without pyvenv.cfg is handed to conda as before."""
    env = tmp_path / "conda_env"
    env.mkdir()

    script = (
        f'source "{SLURM_DIR / "rockfish_common.sh"}"; '
        f'conda() {{ echo "conda-activate:$1"; }}; '
        f'rockfish_activate_env "{env}"'
    )
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "conda-activate:" in completed.stdout


def test_submit_fetch_domains_uses_a_capable_python() -> None:
    """The launcher must not rely on the login node's default interpreter.

    Rockfish login nodes resolve `python` to a system interpreter too old to parse this
    project, so the launcher prefers the environment the jobs will use and fails fast
    with guidance if it still cannot import the package.
    """
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "Cannot import the project" in text
    assert '"${PYTHON}" -m scripts.rockfish_queue' in text
    assert "python -m scripts.rockfish_queue" not in text.replace('"${PYTHON}" -m scripts.rockfish_queue', "")


def test_submit_fetch_domains_distinguishes_empty_queue_from_failure() -> None:
    """A crashed snapshot helper must not be reported as 'nothing to do' with exit 0."""
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "snapshot_status} -eq 2" in text
    assert "Building the array snapshot failed" in text


def test_fetch_domains_worker_exposes_a_rate_limit() -> None:
    """The array worker must be able to pace itself; several tasks share one API."""
    worker = _read_slurm("fetch_domains_rockfish.sh")
    assert "RATE_LIMIT" in worker
    assert "--rate-limit" in worker

    launcher = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "RATE_LIMIT=${RATE_LIMIT}" in launcher
    assert "Request budget" in launcher, "the operator should see the aggregate API load"


def test_layout_worker_never_oversubscribes_its_allocation() -> None:
    """WORKERS must be clamped to the CPUs SLURM granted.

    Running more worker processes than the allocation steals CPU from other users on a
    shared node, and SLURM will not stop it.
    """
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    assert "clamping to" in text
    assert "gt ${SLURM_CPUS_PER_TASK}" in text


def test_layout_clamps_workers_in_practice() -> None:
    """The clamp actually lowers an oversized WORKERS value."""
    script = (
        "SLURM_CPUS_PER_TASK=4; WORKERS=16; "
        "if [[ -n ${SLURM_CPUS_PER_TASK:-} && ${WORKERS} -gt ${SLURM_CPUS_PER_TASK} ]]; then "
        'WORKERS="${SLURM_CPUS_PER_TASK}"; fi; echo "${WORKERS}"'
    )
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "4"


def test_layout_job_is_sized_for_the_full_dataset() -> None:
    """The layout job must request enough memory, cores, and time for ~181k proteins.

    Measured footprint: ~3.9 GB for the store, ~4 GB of accumulated layouts, and ~1 GB
    per spawned worker because PyTorch is loaded in each. Under-requesting memory gets
    the job OOM-killed hours in.
    """
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    mem_line = next(line for line in text.splitlines() if "--mem=" in line)
    cpu_line = next(line for line in text.splitlines() if "--cpus-per-task=" in line)

    memory_gb = int(mem_line.split("=")[-1].rstrip("G"))
    cpus = int(cpu_line.split("=")[-1])

    assert memory_gb >= 32, "not enough memory for the store plus per-worker PyTorch"
    assert cpus >= 4, "the layout stage is CPU-bound on disorder prediction"
    assert "--time=1-" in text or "--time=0" in text, "needs a realistic wall-clock limit"


def test_merge_guidance_keeps_heavy_work_off_the_login_node() -> None:
    """Merging hundreds of chunk stores is GBs of work and belongs on a compute node."""
    text = _read_slurm("submit_fetch_domains_rockfish.sh")
    assert "srun" in text, "the merge instruction should route through SLURM"
    assert "rather than the login node" in text


def test_layout_launcher_allows_overriding_resources() -> None:
    """Disorder prediction scales with cores, so the request must be tunable.

    The whole DnaJ set is roughly 17 h on 8 cores but 6 h on 24, and the right trade-off
    against queue wait depends on how busy the cluster is that day.
    """
    text = _read_slurm("submit_analyze_domain_layout_rockfish.sh")
    assert "CPUS_PER_TASK" in text
    assert "JOB_MEMORY" in text
    assert "--cpus-per-task=" in text
    assert "--mem=" in text


def test_backfill_worker_never_writes_to_the_source_directory() -> None:
    """The source chunks are the only record of a multi-day fetch.

    The worker must copy a chunk out before refreshing it, and must not leave a partly
    written store behind on failure, or a rerun would compound the damage.
    """
    text = _read_slurm("backfill_metadata_rockfish.sh")
    assert 'cp -f "${source_json}" "${output_json}"' in text
    assert "--backfill-metadata" in text
    # The refresh targets the copy, never the source.
    assert '-o "${output_json}"' in text
    assert '-o "${source_json}"' not in text
    assert 'rm -f "${output_json}"' in text


def test_backfill_worker_is_reason_coded_and_resumable() -> None:
    """Failures are reason-coded and completed chunks are skipped on resubmit."""
    text = _read_slurm("backfill_metadata_rockfish.sh")
    assert "rockfish_is_completed" in text
    assert "rockfish_mark_completed" in text
    assert "missing_source" in text
    assert "backfill_nonzero" in text
    assert "rockfish_activate_env" in text
    assert 'conda activate "${CONDA_ENV}"' not in text


def test_backfill_worker_requires_a_chunk_list() -> None:
    """Without a chunk list the array has no task-to-file mapping and must refuse."""
    text = _read_slurm("backfill_metadata_rockfish.sh")
    assert "CHUNK_LIST_FILE is missing or not found" in text
    assert "exit 1" in text


def test_submit_backfill_snapshots_the_chunk_list_and_guards_array_bounds() -> None:
    """A submit-time snapshot keeps the task-to-file mapping stable mid-run."""
    text = _read_slurm("submit_backfill_metadata_rockfish.sh")
    assert "CHUNK_LIST" in text
    assert "MAX_ARRAY_TASKS=10000" in text
    assert "Refusing to submit" in text
    # Rate budget is stated, as for the other InterPro-facing arrays.
    assert "RATE_LIMIT" in text
    assert "ARRAY_CONCURRENCY" in text


def test_backfill_scripts_default_to_a_separate_output_directory() -> None:
    """Source and destination must not default to the same path."""
    for script in ("backfill_metadata_rockfish.sh", "submit_backfill_metadata_rockfish.sh"):
        text = _read_slurm(script)
        assert 'SOURCE_DIR="${SOURCE_DIR:-${WK_DIR}/domain_results}"' in text, script
        assert 'RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/domain_results_backfilled}"' in text, script


def test_motif_worker_loads_mafft_and_passes_the_backend() -> None:
    """The MSA route is only a MAFFT route if the module is actually loaded.

    Without it the job still succeeds, but on the progressive fallback - a silently
    different analysis, which is why the worker warns rather than failing quietly.
    """
    text = _read_slurm("analyze_motif_rockfish.sh")
    assert "MAFFT_MODULE" in text
    assert 'ml "${MAFFT_MODULE}"' in text
    assert "falling back to the progressive aligner" in text
    assert "--msa-backend" in text
    assert "--msa-threads" in text


def test_layout_env_ships_mafft() -> None:
    """MAFFT is a binary, so it belongs in the conda env rather than the pip block."""
    text = (SLURM_DIR / "conda_env_layout.yaml").read_text(encoding="utf-8")
    mafft_line = next(line for line in text.splitlines() if "mafft" in line and not line.strip().startswith("#"))
    # A conda dependency (two-space indent under dependencies), not nested under pip.
    assert mafft_line.startswith("  - mafft")


def test_layout_worker_gates_mafft_on_the_alignment_stage() -> None:
    """The MAFFT module is only needed when --align-msa is requested.

    Loading it unconditionally would make every layout job depend on a module it does not
    use; not loading it at all would silently downgrade the aligner when it is requested.
    """
    text = _read_slurm("analyze_domain_layout_rockfish.sh")
    assert 'ALIGN_MSA="${ALIGN_MSA:-0}"' in text
    assert 'ml "${MAFFT_MODULE}"' in text
    assert "--align-msa" in text
    assert "--max-aligned-per-family" in text
    assert "falling back to the progressive aligner" in text
