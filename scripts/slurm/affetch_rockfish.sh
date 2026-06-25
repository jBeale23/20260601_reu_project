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
#   conda env create -f scripts/slurm/conda_env.yaml
#   conda activate "${HOME}/affetch"
#
# Prepare inputs:
#   fetch-proteins-dnak -o ipr012725_proteins.json
#   extract-uniprot-ids ipr012725_proteins.json -o accessions.txt
#   cp accessions.txt "${WK_DIR}/incomplete_accessions.txt"
#
# Submit (array bounds and log paths set by submit_affetch_rockfish.sh):
#   bash scripts/slurm/submit_affetch_rockfish.sh

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_accessions.txt"
FAILED_LOG="${WK_DIR}/failed_accessions.txt"
OUTPUT_DIR="${WK_DIR}/structures"

# affetch --file-type letters: p=PDB, c=CIF, z=gzip (default in affetch is pcz)
FILE_TYPE="${FILE_TYPE:-pcz}"
MODEL_VERSION="${MODEL_VERSION:-6}"

CONDA_ENV="${CONDA_ENV:-${HOME}/affetch}"

ml anaconda3/2024.02-1
conda activate "${CONDA_ENV}"

if ! command -v affetch > /dev/null 2>&1; then
	printf "affetch not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "Create the environment with: conda env create -f scripts/slurm/conda_env.yaml\n" 1>&2
	exit 1
fi

[[ -f ${INPUT_FILE} ]] || {
	printf "Input file not found: %s\n" "${INPUT_FILE}" 1>&2
	printf "Copy accessions.txt to %s before submitting the array job.\n" "${INPUT_FILE}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${OUTPUT_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"

cd "${WK_DIR}" || exit 1

accession="$(
	sed -n "${SLURM_ARRAY_TASK_ID}"p <(
		comm -23 <(awk 'NF' "${INPUT_FILE}" | sort -u) <(awk 'NF' "${COMPLETION_LOG}" | sort -u)
	)
)"

if [[ -z ${accession} ]]; then
	exit 0
fi

if affetch -o "${OUTPUT_DIR}" -f "${FILE_TYPE}" -m "${MODEL_VERSION}" "${accession}"; then
	printf "%s\n" "${accession}" >> "${COMPLETION_LOG}"
else
	printf "%s\n" "${accession}" >> "${FAILED_LOG}"
fi
