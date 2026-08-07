#!/bin/bash -ue
#SBATCH --job-name="fetch-domains"
#SBATCH --partition=shared
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Fetch full InterPro domain annotations + sequences for a chunk of accessions.
#
# Each array task owns CHUNK_SIZE lines of the submit-time snapshot and writes one
# domain store per chunk. Merge them on a login node with:
#   fetch-protein-domains --merge-stores "${WK_DIR}/domain_results" -o protein_domains.json
#
# Submit:
#   bash scripts/slurm/submit_fetch_domains_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/domain_results}"
ARRAY_QUEUE_FILE="${ARRAY_QUEUE_FILE:-}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
CONCURRENCY="${CONCURRENCY:-4}"
# Requests per second per task. Several tasks run at once, so the aggregate rate seen by
# InterPro is roughly RATE_LIMIT x ARRAY_CONCURRENCY; keep that neighbourly.
RATE_LIMIT="${RATE_LIMIT:-3}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
COMPLETION_LOG="${WK_DIR}/completed_domains.txt"
FAILED_LOG="${WK_DIR}/failed_domains.txt"
COMPLETION_LOCK="${WK_DIR}/.completed_domains.lock"
FAILED_LOCK="${WK_DIR}/.failed_domains.lock"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

ml anaconda3/2024.02-1
rockfish_activate_env "${CONDA_ENV}"

if ! command -v fetch-protein-domains > /dev/null 2>&1; then
	printf "fetch-protein-domains not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure]\"\n" 1>&2
	exit 1
fi

[[ -n ${ARRAY_QUEUE_FILE} && -f ${ARRAY_QUEUE_FILE} ]] || {
	printf "ARRAY_QUEUE_FILE is missing or not found: '%s'\n" "${ARRAY_QUEUE_FILE}" 1>&2
	printf "Re-submit with scripts/slurm/submit_fetch_domains_rockfish.sh.\n" 1>&2
	exit 1
}

# Checkpoints live outside RESULTS_DIR so --merge-stores only sees finished chunks.
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${CHECKPOINT_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${PROJECT_DIR}" || exit 1

start="$(rockfish_chunk_start "${SLURM_ARRAY_TASK_ID}" "${CHUNK_SIZE}")"
marker="$(rockfish_chunk_marker "${start}" "${ARRAY_QUEUE_FILE}" "${CHUNK_SIZE}")"
output_json="${RESULTS_DIR}/protein_domains_${marker}.json"

if rockfish_is_completed "${marker}" "${COMPLETION_LOG}"; then
	exit 0
fi

if [[ -f ${output_json} ]]; then
	rockfish_mark_completed "${marker}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
	exit 0
fi

if fetch-protein-domains \
	--accessions-file "${ARRAY_QUEUE_FILE}" \
	--start "${start}" \
	--count "${CHUNK_SIZE}" \
	--concurrency "${CONCURRENCY}" \
	--rate-limit "${RATE_LIMIT}" \
	--checkpoint "${CHECKPOINT_DIR}/${marker}.json" \
	--quiet \
	-o "${output_json}"; then
	rockfish_mark_completed "${marker}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	printf "fetch-protein-domains failed for chunk starting at line %s (nonzero exit)\n" "${start}" 1>&2
	rockfish_log_failure "${marker}" "${FAILED_LOG}" "${FAILED_LOCK}" "fetch_nonzero" \
		"fetch-protein-domains returned nonzero"
	exit 1
fi
