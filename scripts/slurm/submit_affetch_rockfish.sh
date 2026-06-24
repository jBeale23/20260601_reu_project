#!/bin/bash -ue

# Submit the Rockfish affetch array job with the correct upper bound for pending accessions.
#
# Usage (from project root on Rockfish):
#   bash scripts/slurm/submit_affetch_rockfish.sh

WK_DIR="${HOME}/scr4_sfried3/alphafoldfetch"
PROJECT_DIR="${HOME}/repositories/rockfish-projects/reu_project"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_accessions.txt"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-128}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/affetch_rockfish.sh"

[[ -f ${INPUT_FILE} ]] || {
	printf "Input file not found: %s\n" "${INPUT_FILE}" 1>&2
	printf "Copy accessions.txt to %s before submitting.\n" "${INPUT_FILE}" 1>&2
	exit 1
}

[[ -f ${JOB_SCRIPT} ]] || {
	printf "SLURM job script not found: %s\n" "${JOB_SCRIPT}" 1>&2
	printf "Set PROJECT_DIR or clone the repo to %s before submitting.\n" "${PROJECT_DIR}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}"
touch "${COMPLETION_LOG}"

pending_count="$(
	comm -23 <(sort -u "${INPUT_FILE}") <(sort -u "${COMPLETION_LOG}") | wc -l | awk '{print $1}'
)"

if [[ ${pending_count} -eq 0 ]]; then
	printf "No pending accessions to fetch (input and completion log are in sync).\n" 1>&2
	exit 0
fi

printf "Submitting array job for %s pending accession(s) (concurrency cap: %s).\n" \
	"${pending_count}" "${ARRAY_CONCURRENCY}"

cd "${PROJECT_DIR}" || exit 1
sbatch --array=1-"${pending_count}"%"${ARRAY_CONCURRENCY}" "${JOB_SCRIPT}"
