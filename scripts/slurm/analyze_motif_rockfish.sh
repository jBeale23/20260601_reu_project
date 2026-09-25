#!/bin/bash -ue
#SBATCH --job-name="motif-conservation"
#SBATCH --partition=shared
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Run DnaJ conserved charge / motif window analysis on Rockfish (sequence-based).
#
# Submit:
#   bash scripts/slurm/submit_analyze_motif_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/motif_results}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
MAFFT_MODULE="${MAFFT_MODULE:-}"
MAFFT_VENDOR_DIR="${MAFFT_VENDOR_DIR:-${PROJECT_DIR}/vendor/mafft}"
MSA_BACKEND="${MSA_BACKEND:-auto}"
MSA_THREADS="${MSA_THREADS:-1}"
FETCH_JSON="${FETCH_JSON:-${WK_DIR}/ipr001623_domain_architectures_no_dedup.json}"
MIN_MEMBERS="${MIN_MEMBERS:-3}"
MAX_PER_FAMILY="${MAX_PER_FAMILY:-}"
COMPLETION_LOG="${WK_DIR}/completed_motif.txt"
FAILED_LOG="${WK_DIR}/failed_motif.txt"
COMPLETION_LOCK="${WK_DIR}/.completed_motif.lock"
FAILED_LOCK="${WK_DIR}/.failed_motif.lock"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

ml anaconda3/2024.02-1
# `conda activate` fails outright on a plain venv, which is how the layout environment is
# built when a specific Python is needed for bio-shark.
rockfish_activate_env "${CONDA_ENV}"
# Pin BLAS threading. numpy spawns a thread per core by default, so each worker process
# would claim the whole node's worth of threads; on a shared partition that degrades every
# other job on it. The pipeline sets these for its own workers too, but setting them here
# also covers the parent process and any single-worker run.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Structured domain families are aligned with MAFFT. Without it the analysis still runs,
# but on the weaker progressive aligner, and the summary CSV records that instead.
if rockfish_ensure_mafft "${MAFFT_VENDOR_DIR}" "${MAFFT_MODULE}"; then
	printf "MAFFT: %s\n" "$(command -v mafft)"
else
	printf "WARNING: mafft not found (vendor dir '%s'); falling back to the progressive aligner\n" \
		"${MAFFT_VENDOR_DIR}" 1>&2
fi

if ! command -v analyze-motif-conservation > /dev/null 2>&1; then
	printf "analyze-motif-conservation not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure]\"\n" 1>&2
	exit 1
fi

[[ -f ${FETCH_JSON} ]] || {
	printf "DnaJ fetch JSON not found: %s\n" "${FETCH_JSON}" 1>&2
	printf "Set FETCH_JSON or copy the architecture JSON into WK_DIR.\n" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${PROJECT_DIR}" || exit 1

marker="motif_full_run"
if rockfish_is_completed "${marker}" "${COMPLETION_LOG}"; then
	printf "Motif analysis already marked complete in %s\n" "${COMPLETION_LOG}"
	exit 0
fi

motif_args=(
	"${FETCH_JSON}"
	-o "${RESULTS_DIR}"
	--min-members "${MIN_MEMBERS}"
	--msa-backend "${MSA_BACKEND}"
	--msa-threads "${MSA_THREADS}"
)
if [[ -n ${MAX_PER_FAMILY} ]]; then
	motif_args+=(--max-per-family "${MAX_PER_FAMILY}")
fi
if [[ -n ${HELD_OUT_POCKET_CSV:-} ]]; then
	motif_args+=(--held-out-pocket-csv "${HELD_OUT_POCKET_CSV}")
fi

if analyze-motif-conservation "${motif_args[@]}"; then
	rockfish_mark_completed "${marker}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	printf "analyze-motif-conservation failed (nonzero exit)\n" 1>&2
	rockfish_log_failure "${marker}" "${FAILED_LOG}" "${FAILED_LOCK}" "analyze_nonzero" \
		"analyze-motif-conservation returned nonzero"
	exit 1
fi
