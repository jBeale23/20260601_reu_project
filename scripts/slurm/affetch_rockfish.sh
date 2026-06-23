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
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Download AlphaFold structures on Rockfish using AlphaFoldFetch (affetch).
# https://github.com/mansanlab/alphafoldfetch
#
# Setup (once on Rockfish):
#   conda create -n affetch python=3.12 -y
#   conda activate affetch
#   pip install AlphaFoldFetch
#
# Prepare inputs:
#   fetch-proteins-dnak -o ipr012725_proteins.json
#   python scripts/extract_uniprot_ids.py ipr012725_proteins.json -o accessions.txt
#   cp accessions.txt "${WK_DIR}/incomplete_accessions.txt"
#
# Submit from the project checkout (set array upper bound to pending accession count):
#   cd "${PROJECT_DIR}"
#   N=$(wc -l < "${WK_DIR}/incomplete_accessions.txt")
#   sbatch --array=1-"${N}"%128 scripts/slurm/affetch_rockfish.sh

WK_DIR="${HOME}/scr4_sfried3/alphafoldfetch"
INPUT_FILE="${WK_DIR}/incomplete_accessions.txt"
COMPLETION_LOG="${WK_DIR}/completed_accessions.txt"
OUTPUT_DIR="${WK_DIR}/structures"

# affetch --file-type letters: p=PDB, c=CIF, z=gzip (default in affetch is pcz)
FILE_TYPE="${FILE_TYPE:-pcz}"
MODEL_VERSION="${MODEL_VERSION:-6}"

CONDA_ENV="${CONDA_ENV:-affetch}"

ml anaconda3/2024.02-1
conda activate "${CONDA_ENV}"

mkdir -p "${WK_DIR}" "${OUTPUT_DIR}"
touch "${COMPLETION_LOG}"
if [[ ! -f ${INPUT_FILE} ]]; then
	echo "WARNING: ${INPUT_FILE} not found; creating an empty input file." >&2
	: > "${INPUT_FILE}"
fi

cd "${WK_DIR}" || exit

accession="$(
	sed -n "${SLURM_ARRAY_TASK_ID}"p <(
		comm -23 <(sort -u "${INPUT_FILE}") <(sort -u "${COMPLETION_LOG}")
	)
)"

if [[ -z ${accession} ]]; then
	exit 0
fi

if affetch -o "${OUTPUT_DIR}" -f "${FILE_TYPE}" -m "${MODEL_VERSION}" "${accession}"; then
	printf "%s\n" "${accession}" >> "${COMPLETION_LOG}"
fi
