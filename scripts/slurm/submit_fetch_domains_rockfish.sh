#!/bin/bash -ue

# Submit the Rockfish domain-fetch array job with a fixed per-job accession snapshot.
#
# Usage (from the project root on Rockfish):
#   prepare-rockfish-accessions ipr001623_domain_architectures_no_dedup.json --wk-dir "${WK_DIR}"
#   bash scripts/slurm/submit_fetch_domains_rockfish.sh
#
# Each array task fetches CHUNK_SIZE accessions (default 200) and writes one domain
# store per chunk. Failed chunks are skipped on resubmit unless RETRY_FAILED=1.

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
INPUT_FILE="${INPUT_FILE:-${WK_DIR}/incomplete_accessions.txt}"
COMPLETION_LOG="${WK_DIR}/completed_domains.txt"
FAILED_LOG="${WK_DIR}/failed_domains.txt"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/domain_results}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/fetch_domains_rockfish.sh"
SNAPSHOT_DIR="${WK_DIR}/array_queues"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
CONCURRENCY="${CONCURRENCY:-4}"
RATE_LIMIT="${RATE_LIMIT:-3}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
MAX_ARRAY_TASKS=10000
RETRY_FAILED="${RETRY_FAILED:-0}"

[[ -f ${INPUT_FILE} ]] || {
	printf "Input file not found: %s\n" "${INPUT_FILE}" 1>&2
	printf "Run: prepare-rockfish-accessions <fetch.json> --wk-dir %s\n" "${WK_DIR}" 1>&2
	exit 1
}

[[ -f ${JOB_SCRIPT} ]] || {
	printf "SLURM job script not found: %s\n" "${JOB_SCRIPT}" 1>&2
	printf "Set PROJECT_DIR or clone the repo to %s before submitting.\n" "${PROJECT_DIR}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${SNAPSHOT_DIR}" "${WK_DIR}/logs"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${PROJECT_DIR}" || exit 1

# The launcher runs queue helpers on the login node, where the default `python` is the
# ancient system interpreter that cannot even parse this project. Prefer the environment
# the jobs themselves will use.
if [[ -z ${PYTHON:-} ]]; then
	if [[ -x ${CONDA_ENV}/bin/python ]]; then
		PYTHON="${CONDA_ENV}/bin/python"
	else
		PYTHON="python"
	fi
fi

if ! "${PYTHON}" -c "import scripts.rockfish_queue" > /dev/null 2>&1; then
	printf "Cannot import the project with '%s'.\n" "${PYTHON}" 1>&2
	printf "Set CONDA_ENV to the environment holding this project, or set PYTHON directly.\n" 1>&2
	exit 1
fi

"${PYTHON}" -m scripts.rockfish_queue dedupe-file "${INPUT_FILE}"

SNAPSHOT="${SNAPSHOT_DIR}/domains_$(date +%Y%m%d_%H%M%S).txt"

# The snapshot is always the FULL deduped accession list. Capping it would strand every
# accession past the cap: completion is tracked per chunk, so a later submission would
# rebuild the same truncated queue and never reach the remainder. The array-bound check
# below keeps the run legal by requiring a large enough CHUNK_SIZE instead.
snapshot_args=(
	write-snapshot
	--input "${INPUT_FILE}"
	--completed "${COMPLETION_LOG}"
	-o "${SNAPSHOT}"
	--limit "$(awk 'NF' "${INPUT_FILE}" | wc -l)"
)

set +e
"${PYTHON}" -m scripts.rockfish_queue "${snapshot_args[@]}"
snapshot_status=$?
set -e

# write-snapshot exits 2 when nothing is pending; any other nonzero status is a real
# failure and must not be reported as "nothing to do".
if [[ ${snapshot_status} -eq 2 ]]; then
	printf "No accessions to queue for domain fetching.\n" 1>&2
	exit 0
fi
if [[ ${snapshot_status} -ne 0 ]]; then
	printf "Building the array snapshot failed (exit %s); not submitting.\n" "${snapshot_status}" 1>&2
	exit 1
fi

accession_count="$(wc -l < "${SNAPSHOT}" | awk '{print $1}')"
task_count="$(((accession_count + CHUNK_SIZE - 1) / CHUNK_SIZE))"

if [[ ${task_count} -lt 1 ]]; then
	printf "Snapshot is empty: %s\n" "${SNAPSHOT}" 1>&2
	exit 1
fi

# sbatch rejects an array larger than MaxArraySize, so fail here with the fix rather
# than after the queue has been written.
if [[ ${task_count} -gt ${MAX_ARRAY_TASKS} ]]; then
	required_chunk="$(((accession_count + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS))"
	printf "Refusing to submit %s array tasks (max %s).\n" "${task_count}" "${MAX_ARRAY_TASKS}" 1>&2
	printf "Raise CHUNK_SIZE to at least %s: CHUNK_SIZE=%s bash %s\n" \
		"${required_chunk}" "${required_chunk}" "scripts/slurm/submit_fetch_domains_rockfish.sh" 1>&2
	exit 1
fi

printf "Submitting domain-fetch array: %s accession(s) in %s chunk(s) of %s (concurrency cap: %s).\n" \
	"${accession_count}" "${task_count}" "${CHUNK_SIZE}" "${ARRAY_CONCURRENCY}"
printf "Request budget: %s req/s per task x %s tasks = up to %s req/s at InterPro.\n" \
	"${RATE_LIMIT}" "${ARRAY_CONCURRENCY}" "$((RATE_LIMIT * ARRAY_CONCURRENCY))"
if [[ ${RETRY_FAILED} == "1" ]]; then
	printf "RETRY_FAILED=1: clearing %s so failed chunks are re-queued.\n" "${FAILED_LOG}"
	: > "${FAILED_LOG}"
fi

sbatch_args=(
	--output="${WK_DIR}/logs/domains_%A_%a.out"
	--error="${WK_DIR}/logs/domains_%A_%a.err"
	--array=1-"${task_count}"%"${ARRAY_CONCURRENCY}"
)
sbatch_export="ALL,ARRAY_QUEUE_FILE=${SNAPSHOT},PROJECT_DIR=${PROJECT_DIR},WK_DIR=${WK_DIR}"
sbatch_export+=",RESULTS_DIR=${RESULTS_DIR},CHUNK_SIZE=${CHUNK_SIZE},CONCURRENCY=${CONCURRENCY}"
sbatch_export+=",RATE_LIMIT=${RATE_LIMIT}"
sbatch_export+=",CONDA_ENV=${CONDA_ENV}"
sbatch_args+=(--export="${sbatch_export}")

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Snapshot: %s\n" "${SNAPSHOT}"
printf "Track logs under %s/logs/domains_%s_<taskid>.{out,err}\n" "${WK_DIR}" "${job_id}"
printf "\nAfter the array completes, merge the per-chunk stores. The merged store is several\n"
printf "GB in memory, so run it on a compute node rather than the login node:\n"
printf "  srun --account=%s --partition=shared --time=00:30:00 --mem=16G \\\n" "${SLURM_ACCOUNT:-sfried3}"
printf "    fetch-protein-domains --merge-stores \"%s\" -o \"%s/protein_domains.json\"\n" \
	"${RESULTS_DIR}" "${WK_DIR}"
printf "  bash scripts/slurm/submit_analyze_domain_layout_rockfish.sh\n"
