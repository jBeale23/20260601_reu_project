#!/bin/bash -ue
#SBATCH --job-name="domain-layout"
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Run the dual-layer domain-layout analysis (metapredict + SHARK) on a merged domain store.
#
# Sizing for the full ~181k-protein DnaJ set, from measurements rather than guesswork:
#   domain store in RAM  ~3.9 GB   (22 KB/protein)
#   accumulated layouts  ~4.0 GB   (14-23 KB/protein)
#   each worker process  ~1.0 GB   (PyTorch is loaded per spawned worker)
# 12 workers therefore peak near 20 GB; 64 GB leaves comfortable headroom.
#
# Submit:
#   bash scripts/slurm/submit_analyze_domain_layout_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/layout_results}"
DOMAIN_STORE="${DOMAIN_STORE:-${WK_DIR}/protein_domains.json}"
CONDA_ENV="${CONDA_ENV:-${HOME}/layout}"
DISORDER_BACKEND="${DISORDER_BACKEND:-auto}"
SHARK_BACKEND="${SHARK_BACKEND:-auto}"
MAFFT_MODULE="${MAFFT_MODULE:-mafft/7.525}"
# Off by default: aligning the MSA-routed subFASTAs is a second substantial compute stage
# on top of the per-protein analysis. Set ALIGN_MSA=1 to run it in the same job.
ALIGN_MSA="${ALIGN_MSA:-0}"
MAX_ALIGNED_PER_FAMILY="${MAX_ALIGNED_PER_FAMILY:-2000}"
MAX_PROTEINS="${MAX_PROTEINS:-}"
# Default to the cores SLURM granted this job, and never exceed them: running more
# worker processes than the allocation oversubscribes a node shared with other users.
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
if [[ -n ${SLURM_CPUS_PER_TASK:-} && ${WORKERS} -gt ${SLURM_CPUS_PER_TASK} ]]; then
	printf "WORKERS=%s exceeds the %s CPU(s) allocated; clamping to %s.\n" \
		"${WORKERS}" "${SLURM_CPUS_PER_TASK}" "${SLURM_CPUS_PER_TASK}" 1>&2
	WORKERS="${SLURM_CPUS_PER_TASK}"
fi
COMPLETION_LOG="${WK_DIR}/completed_layout.txt"
FAILED_LOG="${WK_DIR}/failed_layout.txt"
COMPLETION_LOCK="${WK_DIR}/.completed_layout.lock"
FAILED_LOCK="${WK_DIR}/.failed_layout.lock"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

ml anaconda3/2024.02-1
if [[ ${ALIGN_MSA} == "1" ]]; then
	# Only needed for --align-msa; without it the run still succeeds, but on the weaker
	# progressive fallback, which the summary would then record.
	ml "${MAFFT_MODULE}" 2> /dev/null || printf "WARNING: could not load %s; falling back to the progressive aligner\n" "${MAFFT_MODULE}" 1>&2
fi
rockfish_activate_env "${CONDA_ENV}"

if ! command -v analyze-domain-layout > /dev/null 2>&1; then
	printf "analyze-domain-layout not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure,layout]\"\n" 1>&2
	exit 1
fi

[[ -f ${DOMAIN_STORE} ]] || {
	printf "Domain store not found: %s\n" "${DOMAIN_STORE}" 1>&2
	printf "Run the domain fetch array first, then merge:\n" 1>&2
	printf "  fetch-protein-domains --merge-stores \"%s/domain_results\" -o \"%s\"\n" "${WK_DIR}" "${DOMAIN_STORE}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${PROJECT_DIR}" || exit 1

# Record which backends are actually active so results are never misattributed.
analyze-domain-layout --show-backends \
	--disorder-backend "${DISORDER_BACKEND}" \
	--shark-backend "${SHARK_BACKEND}"

marker="layout_full_run"
if rockfish_is_completed "${marker}" "${COMPLETION_LOG}"; then
	printf "Domain layout already marked complete in %s\n" "${COMPLETION_LOG}"
	exit 0
fi

layout_args=(
	"${DOMAIN_STORE}"
	-o "${RESULTS_DIR}"
	--disorder-backend "${DISORDER_BACKEND}"
	--shark-backend "${SHARK_BACKEND}"
	--workers "${WORKERS}"
)
if [[ -n ${MAX_PROTEINS} ]]; then
	layout_args+=(--max-proteins "${MAX_PROTEINS}")
fi
if [[ ${ALIGN_MSA} == "1" ]]; then
	layout_args+=(--align-msa --max-aligned-per-family "${MAX_ALIGNED_PER_FAMILY}")
fi

if analyze-domain-layout "${layout_args[@]}"; then
	rockfish_mark_completed "${marker}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	printf "analyze-domain-layout failed (nonzero exit)\n" 1>&2
	rockfish_log_failure "${marker}" "${FAILED_LOG}" "${FAILED_LOCK}" "analyze_nonzero" \
		"analyze-domain-layout returned nonzero"
	exit 1
fi
