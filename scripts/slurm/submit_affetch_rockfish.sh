#!/bin/bash -ue

# Submit the Rockfish affetch array job with a fixed per-job accession snapshot.
#
# Usage (from project root on Rockfish):
#   prepare-rockfish-accessions ipr012725_proteins.json --wk-dir "${WK_DIR}"
#   bash scripts/slurm/submit_affetch_rockfish.sh

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_accessions.txt"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-128}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/affetch_rockfish.sh"
SNAPSHOT_DIR="${WK_DIR}/array_queues"
MAX_ARRAY_TASKS=10000

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

mkdir -p "${WK_DIR}" "${SNAPSHOT_DIR}"
touch "${COMPLETION_LOG}"

cd "${PROJECT_DIR}" || exit 1

python -m scripts.rockfish_queue dedupe-file "${INPUT_FILE}"

input_count="$(
	awk 'NF' "${INPUT_FILE}" | sort -u | wc -l | awk '{print $1}'
)"

if [[ ${input_count} -eq 0 ]]; then
	printf "Input file contains no accessions: %s\n" "${INPUT_FILE}" 1>&2
	exit 1
fi

pending_count="$(
	comm -23 <(awk 'NF' "${INPUT_FILE}" | sort -u) <(awk 'NF' "${COMPLETION_LOG}" | sort -u) | wc -l | awk '{print $1}'
)"

if [[ ${pending_count} -eq 0 ]]; then
	printf "All %s accession(s) in %s are already listed in %s.\n" \
		"${input_count}" "${INPUT_FILE}" "${COMPLETION_LOG}" 1>&2
	exit 0
fi

SNAPSHOT="${SNAPSHOT_DIR}/affetch_$(date +%Y%m%d_%H%M%S).txt"
python -m scripts.rockfish_queue write-snapshot \
	--input "${INPUT_FILE}" \
	--completed "${COMPLETION_LOG}" \
	-o "${SNAPSHOT}" \
	--limit "${MAX_ARRAY_TASKS}"

pending_count="$(wc -l < "${SNAPSHOT}" | awk '{print $1}')"

mkdir -p "${WK_DIR}/logs"

printf "Submitting array job for %s accession(s) from snapshot %s (concurrency cap: %s).\n" \
	"${pending_count}" "${SNAPSHOT}" "${ARRAY_CONCURRENCY}"

sbatch_args=(
	--output="${WK_DIR}/logs/affetch_%A_%a.out"
	--error="${WK_DIR}/logs/affetch_%A_%a.err"
	--array=1-"${pending_count}"%"${ARRAY_CONCURRENCY}"
)
sbatch_export="ALL,ARRAY_QUEUE_FILE=${SNAPSHOT},PROJECT_DIR=${PROJECT_DIR},WK_DIR=${WK_DIR}"
sbatch_args+=(--export="${sbatch_export}")

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Snapshot: %s\n" "${SNAPSHOT}"
printf "Track logs under %s/logs/affetch_%s_<taskid>.{out,err}\n" "${WK_DIR}" "${job_id}"
