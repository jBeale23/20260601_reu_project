#!/bin/bash -ue

# Submit the Rockfish pocket-charge array job with a fixed per-job accession snapshot.
#
# Usage (from project root on Rockfish):
#   prepare-rockfish-accessions ipr012725_proteins.json --wk-dir "${WK_DIR}"
#   bash scripts/slurm/submit_analyze_pocket_rockfish.sh
#
# Failed accessions in failed_pocket.txt are skipped by default.
# To retry them: RETRY_FAILED=1 bash scripts/slurm/submit_analyze_pocket_rockfish.sh
# PDB preflight is on by default (skips CIF-only / missing structures).

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_pocket.txt"
FAILED_LOG="${WK_DIR}/failed_pocket.txt"
STRUCTURES_DIR="${STRUCTURES_DIR:-${WK_DIR}/structures}"
RESULTS_DIR="${WK_DIR}/pocket_results"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-128}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/analyze_pocket_rockfish.sh"
SNAPSHOT_DIR="${WK_DIR}/array_queues"
MAX_ARRAY_TASKS=10000
MODEL_VERSION="${MODEL_VERSION:-6}"
REQUIRE_PDB="${REQUIRE_PDB:-1}"
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

mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${SNAPSHOT_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${PROJECT_DIR}" || exit 1

python -m scripts.rockfish_queue dedupe-file "${INPUT_FILE}"

input_count="$(
	awk 'NF' "${INPUT_FILE}" | sort -u | wc -l | awk '{print $1}'
)"

if [[ ${input_count} -eq 0 ]]; then
	printf "Input file contains no accessions: %s\n" "${INPUT_FILE}" 1>&2
	exit 1
fi

SNAPSHOT="${SNAPSHOT_DIR}/pocket_$(date +%Y%m%d_%H%M%S).txt"
SKIPPED_LOG="${SNAPSHOT_DIR}/pocket_skipped_no_pdb_$(date +%Y%m%d_%H%M%S).txt"

snapshot_args=(
	write-snapshot
	--input "${INPUT_FILE}"
	--completed "${COMPLETION_LOG}"
	--failed "${FAILED_LOG}"
	-o "${SNAPSHOT}"
	--limit "${MAX_ARRAY_TASKS}"
)

if [[ ${RETRY_FAILED} == "1" ]]; then
	snapshot_args+=(--retry-failed)
fi

if [[ ${REQUIRE_PDB} == "1" ]]; then
	snapshot_args+=(
		--require-pdb
		--structures-dir "${STRUCTURES_DIR}"
		--model-version "${MODEL_VERSION}"
		--skipped-output "${SKIPPED_LOG}"
	)
fi

set +e
python -m scripts.rockfish_queue "${snapshot_args[@]}"
snapshot_status=$?
set -e

if [[ ${snapshot_status} -ne 0 ]]; then
	printf "No pending pocket-charge accessions to submit"
	if [[ ${REQUIRE_PDB} == "1" ]]; then
		printf " (after excluding completed/failed and accessions without PDB)"
	fi
	printf ".\n" 1>&2
	if [[ -f ${SKIPPED_LOG} ]]; then
		printf "Skipped (no PDB) log: %s\n" "${SKIPPED_LOG}" 1>&2
	fi
	exit 0
fi

pending_count="$(wc -l < "${SNAPSHOT}" | awk '{print $1}')"

mkdir -p "${WK_DIR}/logs"

printf "Submitting pocket-charge array job for %s accession(s) from snapshot %s (concurrency cap: %s).\n" \
	"${pending_count}" "${SNAPSHOT}" "${ARRAY_CONCURRENCY}"
if [[ -f ${SKIPPED_LOG} ]]; then
	printf "PDB preflight skipped log: %s\n" "${SKIPPED_LOG}"
fi

sbatch_args=(
	--output="${WK_DIR}/logs/pocket_%A_%a.out"
	--error="${WK_DIR}/logs/pocket_%A_%a.err"
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
printf "Track logs under %s/logs/pocket_%s_<taskid>.{out,err}\n" "${WK_DIR}" "${job_id}"
printf "\nAfter the array completes, merge results on a login node:\n"
printf "  conda activate \"%s\"\n" "${CONDA_ENV:-${HOME}/pocket}"
printf "  cd \"%s\"\n" "${PROJECT_DIR}"
printf "  analyze-pocket-charge --merge-results \"%s\" --output-dir \"%s\" --min-mapping-confidence high\n" \
	"${RESULTS_DIR}" "${RESULTS_DIR}"
printf "  merge-features ipr012725_proteins.json \\\n"
printf "    --pocket-csv \"%s/pocket_charge_summary.csv\" \\\n" "${RESULTS_DIR}"
printf "    -o \"%s/merged_features/dnak_with_pocket.csv\"\n" "${WK_DIR}"
