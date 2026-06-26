#!/bin/bash -ue
#SBATCH --job-name="pocket-charge"
#SBATCH --partition=shared
#SBATCH --time=00-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --array=1-10000%128
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Run analyze-pocket-charge on AlphaFold structures downloaded via affetch.
#
# Submit (writes a fixed array snapshot):
#   bash scripts/slurm/submit_analyze_pocket_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_pocket.txt"
FAILED_LOG="${WK_DIR}/failed_pocket.txt"
COMPLETION_LOCK="${WK_DIR}/.completed_pocket.lock"
FAILED_LOCK="${WK_DIR}/.failed_pocket.lock"
ARRAY_QUEUE_FILE="${ARRAY_QUEUE_FILE:-}"
STRUCTURES_DIR="${STRUCTURES_DIR:-${WK_DIR}/structures}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/pocket_results}"
REFERENCE_ACCESSION="${REFERENCE_ACCESSION:-P0A6Y8}"
MODEL_VERSION="${MODEL_VERSION:-6}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"

POCKET_REF="${PROJECT_DIR}/data/pocket_refs/dnak_sbd_pocket.yaml"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

REFERENCE_STRUCTURE="$(rockfish_resolve_pdb_structure "${REFERENCE_ACCESSION}" "${STRUCTURES_DIR}" "${MODEL_VERSION}")"

ml anaconda3/2024.02-1
conda activate "${CONDA_ENV}"

if ! command -v analyze-pocket-charge > /dev/null 2>&1; then
	printf "analyze-pocket-charge not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure]\"\n" 1>&2
	exit 1
fi

[[ -f ${INPUT_FILE} ]] || {
	printf "Input file not found: %s\n" "${INPUT_FILE}" 1>&2
	printf "Run: prepare-rockfish-accessions <fetch.json> --wk-dir %s\n" "${WK_DIR}" 1>&2
	exit 1
}

[[ -f ${POCKET_REF} ]] || {
	printf "Pocket reference not found: %s\n" "${POCKET_REF}" 1>&2
	exit 1
}

[[ -n ${REFERENCE_STRUCTURE} && -f ${REFERENCE_STRUCTURE} ]] || {
	printf "Reference structure not found for %s under %s (model v%s; PDB required for pocket charge)\n" \
		"${REFERENCE_ACCESSION}" "${STRUCTURES_DIR}" "${MODEL_VERSION}" 1>&2
	printf 'Ensure %s was downloaded via affetch before submitting.\n' "${REFERENCE_ACCESSION}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${WK_DIR}" || exit 1

accession="$(rockfish_read_task_accession "${SLURM_ARRAY_TASK_ID}" "${ARRAY_QUEUE_FILE}")" || exit 1

if [[ -z ${accession} ]]; then
	exit 0
fi

if rockfish_is_completed "${accession}" "${COMPLETION_LOG}"; then
	exit 0
fi

output_json="${RESULTS_DIR}/pocket_charge_${accession}.json"
if [[ -f ${output_json} ]]; then
	rockfish_mark_completed "${accession}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
	exit 0
fi

structure="$(rockfish_resolve_pdb_structure "${accession}" "${STRUCTURES_DIR}" "${MODEL_VERSION}")"
if [[ -z ${structure} || ! -f ${structure} ]]; then
	printf "Structure not found for %s under %s (model v%s; PDB required for pocket charge)\n" "${accession}" "${STRUCTURES_DIR}" "${MODEL_VERSION}" 1>&2
	rockfish_log_failure "${accession}" "${FAILED_LOG}" "${FAILED_LOCK}"
	exit 1
fi

if analyze-pocket-charge "${structure}" \
	--reference-pocket "${POCKET_REF}" \
	--reference-structure "${REFERENCE_STRUCTURE}" \
	-o "${output_json}"; then
	rockfish_mark_completed "${accession}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	rockfish_log_failure "${accession}" "${FAILED_LOG}" "${FAILED_LOCK}"
	exit 1
fi
