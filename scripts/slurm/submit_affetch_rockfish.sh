#!/bin/bash -ue

# Submit the Rockfish affetch array job with the correct upper bound for pending accessions.
#
# Usage (from project root on Rockfish):
#   bash scripts/slurm/submit_affetch_rockfish.sh

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
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

mkdir -p "${WK_DIR}/logs"

# This is done as slurm on rockfish is configured to allow a maximum array job size of 10,000
pending_count=$(("${pending_count}" > 10000 ? 10000 : "${pending_count}"))

printf "Submitting array job for %s pending accession(s) (concurrency cap: %s).\n" \
	"${pending_count}" "${ARRAY_CONCURRENCY}"

cd "${PROJECT_DIR}" || exit 1

sbatch_args=(
	--output="${WK_DIR}/logs/affetch_%A_%a.out"
	--error="${WK_DIR}/logs/affetch_%A_%a.err"
	--array=1-"${pending_count}"%"${ARRAY_CONCURRENCY}"
)

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Track logs under %s/logs/affetch_%s_<taskid>.{out,err}\n" "${WK_DIR}" "${job_id}"
