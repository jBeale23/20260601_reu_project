#!/bin/bash -ue
#SBATCH --job-name="backfill-metadata"
#SBATCH --partition=shared
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Refresh protein metadata (the UniProt fragment flag) on already-fetched chunk stores.
#
# A store fetched before is_fragment was recorded has every domain match it needs; only
# the per-protein metadata is stale. Re-running the whole fetch to gain one boolean would
# repeat the expensive half of the work for nothing, so this re-requests the protein
# endpoint alone - one request per accession, no entry queries.
#
# Each array task owns one chunk store. The source directory is never written to: chunks
# are copied into RESULTS_DIR first, so a failed task can be re-run and the originals
# stay intact.
#
# Merge afterwards on a login node with:
#   fetch-protein-domains --merge-stores "${RESULTS_DIR}" -o protein_domains.json
#
# Submit:
#   bash scripts/slurm/submit_backfill_metadata_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
SOURCE_DIR="${SOURCE_DIR:-${WK_DIR}/domain_results}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/domain_results_backfilled}"
CHUNK_LIST_FILE="${CHUNK_LIST_FILE:-}"
CONCURRENCY="${CONCURRENCY:-4}"
# Requests per second per task. Several tasks run at once, so the aggregate rate seen by
# InterPro is roughly RATE_LIMIT x ARRAY_CONCURRENCY; keep that neighbourly.
RATE_LIMIT="${RATE_LIMIT:-3}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
COMPLETION_LOG="${WK_DIR}/completed_backfill.txt"
FAILED_LOG="${WK_DIR}/failed_backfill.txt"
COMPLETION_LOCK="${WK_DIR}/.completed_backfill.lock"
FAILED_LOCK="${WK_DIR}/.failed_backfill.lock"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

ml anaconda3/2024.02-1
rockfish_activate_env "${CONDA_ENV}"

if ! command -v fetch-protein-domains > /dev/null 2>&1; then
	printf "fetch-protein-domains not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure]\"\n" 1>&2
	exit 1
fi

[[ -n ${CHUNK_LIST_FILE} && -f ${CHUNK_LIST_FILE} ]] || {
	printf "CHUNK_LIST_FILE is missing or not found: '%s'\n" "${CHUNK_LIST_FILE}" 1>&2
	printf "Re-submit with scripts/slurm/submit_backfill_metadata_rockfish.sh.\n" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}"
touch "${COMPLETION_LOG}" "${FAILED_LOG}"
cd "${PROJECT_DIR}" || exit 1

# One chunk store per array task; task IDs are 1-based, the file list is 0-based.
chunk_name="$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "${CHUNK_LIST_FILE}")"
[[ -n ${chunk_name} ]] || {
	printf "No chunk on line %s of %s; nothing to do.\n" "${SLURM_ARRAY_TASK_ID}" "${CHUNK_LIST_FILE}" 1>&2
	exit 0
}

source_json="${SOURCE_DIR}/${chunk_name}"
output_json="${RESULTS_DIR}/${chunk_name}"
marker="backfill_${chunk_name}"

if rockfish_is_completed "${marker}" "${COMPLETION_LOG}"; then
	exit 0
fi

[[ -f ${source_json} ]] || {
	printf "Source chunk store not found: %s\n" "${source_json}" 1>&2
	rockfish_log_failure "${marker}" "${FAILED_LOG}" "${FAILED_LOCK}" "missing_source" \
		"source chunk store not found"
	exit 1
}

# Copy first, then refresh the copy in place. The source directory is the only record of
# a fetch that took days; a partially written store must never be able to overwrite it.
cp -f "${source_json}" "${output_json}"

if fetch-protein-domains \
	--backfill-metadata \
	--concurrency "${CONCURRENCY}" \
	--rate-limit "${RATE_LIMIT}" \
	--quiet \
	-o "${output_json}"; then
	rockfish_mark_completed "${marker}" "${COMPLETION_LOG}" "${COMPLETION_LOCK}"
else
	printf "backfill failed for chunk %s (nonzero exit)\n" "${chunk_name}" 1>&2
	rm -f "${output_json}"
	rockfish_log_failure "${marker}" "${FAILED_LOG}" "${FAILED_LOCK}" "backfill_nonzero" \
		"fetch-protein-domains --backfill-metadata returned nonzero"
	exit 1
fi
