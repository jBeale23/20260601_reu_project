#!/bin/bash -ue

# Submit the metadata-backfill array: refresh the UniProt fragment flag on chunk stores
# that were fetched before it was recorded.
#
# Usage (from the project root on Rockfish):
#   bash scripts/slurm/submit_backfill_metadata_rockfish.sh
#
# One array task per chunk store. Domain matches are not re-requested, so this costs one
# protein-endpoint request per accession - roughly half a fetch, and none of the paging.
# Completed chunks are skipped on resubmit unless RETRY_FAILED=1.

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
SOURCE_DIR="${SOURCE_DIR:-${WK_DIR}/domain_results}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/domain_results_backfilled}"
COMPLETION_LOG="${WK_DIR}/completed_backfill.txt"
FAILED_LOG="${WK_DIR}/failed_backfill.txt"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/backfill_metadata_rockfish.sh"
SNAPSHOT_DIR="${WK_DIR}/array_queues"
CONCURRENCY="${CONCURRENCY:-4}"
RATE_LIMIT="${RATE_LIMIT:-3}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
MAX_ARRAY_TASKS=10000
RETRY_FAILED="${RETRY_FAILED:-0}"

[[ -d ${SOURCE_DIR} ]] || {
	printf "Source chunk directory not found: %s\n" "${SOURCE_DIR}" 1>&2
	printf "Set SOURCE_DIR to the directory holding the per-chunk domain stores.\n" 1>&2
	exit 1
}

[[ -f ${JOB_SCRIPT} ]] || {
	printf "SLURM job script not found: %s\n" "${JOB_SCRIPT}" 1>&2
	printf "Set PROJECT_DIR or clone the repo to %s before submitting.\n" "${PROJECT_DIR}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${SNAPSHOT_DIR}" "${WK_DIR}/logs"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

if [[ ${RETRY_FAILED} == "1" ]]; then
	printf "RETRY_FAILED=1: clearing %s so failed chunks are re-queued.\n" "${FAILED_LOG}"
	: > "${FAILED_LOG}"
fi

# The chunk list is a submit-time snapshot, so a chunk written while the array is running
# cannot shift the task-to-file mapping midway and send two tasks at the same store.
CHUNK_LIST="${SNAPSHOT_DIR}/backfill_$(date +%Y%m%d_%H%M%S).txt"
find "${SOURCE_DIR}" -maxdepth 1 -name '*.json' -printf '%f\n' | sort > "${CHUNK_LIST}"

task_count="$(wc -l < "${CHUNK_LIST}" | awk '{print $1}')"
if [[ ${task_count} -lt 1 ]]; then
	printf "No chunk stores found in %s; nothing to back-fill.\n" "${SOURCE_DIR}" 1>&2
	rm -f "${CHUNK_LIST}"
	exit 0
fi

if [[ ${task_count} -gt ${MAX_ARRAY_TASKS} ]]; then
	printf "Refusing to submit %s array tasks (max %s).\n" "${task_count}" "${MAX_ARRAY_TASKS}" 1>&2
	printf "Split %s across two runs, or merge the chunk stores first.\n" "${SOURCE_DIR}" 1>&2
	exit 1
fi

printf "Submitting metadata backfill: %s chunk store(s) (concurrency cap: %s).\n" \
	"${task_count}" "${ARRAY_CONCURRENCY}"
printf "Request budget: %s req/s per task x %s tasks = up to %s req/s at InterPro.\n" \
	"${RATE_LIMIT}" "${ARRAY_CONCURRENCY}" "$((RATE_LIMIT * ARRAY_CONCURRENCY))"
printf "Source (never written to): %s\nOutput: %s\n" "${SOURCE_DIR}" "${RESULTS_DIR}"

sbatch_args=(
	--output="${WK_DIR}/logs/backfill_%A_%a.out"
	--error="${WK_DIR}/logs/backfill_%A_%a.err"
	--array=1-"${task_count}"%"${ARRAY_CONCURRENCY}"
)
sbatch_export="ALL,CHUNK_LIST_FILE=${CHUNK_LIST},PROJECT_DIR=${PROJECT_DIR},WK_DIR=${WK_DIR}"
sbatch_export+=",SOURCE_DIR=${SOURCE_DIR},RESULTS_DIR=${RESULTS_DIR}"
sbatch_export+=",CONCURRENCY=${CONCURRENCY},RATE_LIMIT=${RATE_LIMIT}"
sbatch_export+=",CONDA_ENV=${CONDA_ENV}"
sbatch_args+=(--export="${sbatch_export}")

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Chunk list: %s\n" "${CHUNK_LIST}"
printf "Track logs under %s/logs/backfill_%s_<taskid>.{out,err}\n" "${WK_DIR}" "${job_id}"
printf "\nAfter the array completes, merge the back-filled stores on a compute node:\n"
printf "  srun --account=%s --partition=shared --time=00:30:00 --mem=16G \\\n" "${SLURM_ACCOUNT:-sfried3}"
printf "    fetch-protein-domains --merge-stores \"%s\" -o \"%s/protein_domains.json\"\n" \
	"${RESULTS_DIR}" "${WK_DIR}"
