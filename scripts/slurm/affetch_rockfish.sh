#!/bin/bash -ue
#SBATCH --job-name="affetch"
#SBATCH --partition=shared
#SBATCH --time=00-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --array=1-10000%128
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Download AlphaFold structures on Rockfish using AlphaFoldFetch (affetch).
# https://github.com/mansanlab/alphafoldfetch
#
# Setup (once on Rockfish):
#   conda env create -f scripts/slurm/conda_env.yaml -p "${HOME}/affetch"
#   conda activate "${HOME}/affetch"
#
# Prepare inputs (run once after each fetch; dedupes and validates counts):
#   prepare-rockfish-accessions ipr012725_proteins.json --wk-dir "${WK_DIR}"
#
# Submit (writes a fixed array snapshot; do not run affetch_rockfish.sh directly):
#   bash scripts/slurm/submit_affetch_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_accessions.txt"
FAILED_LOG="${WK_DIR}/failed_accessions.txt"
OUTPUT_DIR="${WK_DIR}/structures"
COMPLETION_LOCK="${WK_DIR}/.completed_accessions.lock"
FAILED_LOCK="${WK_DIR}/.failed_accessions.lock"
ARRAY_QUEUE_FILE="${ARRAY_QUEUE_FILE:-}"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

FILE_TYPE="${FILE_TYPE:-pcz}"
MODEL_VERSION="${MODEL_VERSION:-6}"
CONDA_ENV="${CONDA_ENV:-${HOME}/affetch}"

ml anaconda3/2024.02-1
conda activate "${CONDA_ENV}"

if ! command -v affetch > /dev/null 2>&1; then
	printf "affetch not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf 'Create the environment with: conda env create -f scripts/slurm/conda_env.yaml -p "%s/affetch"\n' "${HOME}" 1>&2
	exit 1
fi

[[ -f ${INPUT_FILE} ]] || {
	printf "Input file not found: %s\n" "${INPUT_FILE}" 1>&2
	printf "Run: prepare-rockfish-accessions <fetch.json> --wk-dir %s\n" "${WK_DIR}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${OUTPUT_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${WK_DIR}" || exit 1

accession="$(rockfish_read_task_accession "${SLURM_ARRAY_TASK_ID}" "${ARRAY_QUEUE_FILE}")" || exit 1

if [[ -z ${accession} ]]; then
	exit 0
fi

if rockfish_is_completed "${accession}" "${COMPLETION_LOG}"; then
	exit 0
fi

if rockfish_structure_exists "${accession}" "${OUTPUT_DIR}" "${MODEL_VERSION}"; then
	rockfish_mark_completed "${accession}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
	exit 0
fi

if affetch -o "${OUTPUT_DIR}" -f "${FILE_TYPE}" -m "${MODEL_VERSION}" "${accession}"; then
	rockfish_mark_completed "${accession}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	rockfish_log_failure "${accession}" "${FAILED_LOG}" "${FAILED_LOCK}"
	exit 1
fi
