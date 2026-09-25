#!/bin/bash -ue
#SBATCH --job-name="fetch-hsp70"
#SBATCH --partition=shared
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --mail-user=jbeale3@jh.edu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Fetch every protein carrying IPR013126 (Hsp70 family) from InterPro.
#
# This is one sequential paginated crawl against a rate-limited API, not a parallel
# workload, so it is a single task rather than an array: the pagination cursor is a single
# piece of state and splitting it would mean guessing page boundaries in advance.
#
# The crawl is resumable. It checkpoints its cursor and the proteins fetched so far, so a
# job that dies partway costs only the pages it had not yet reached. An earlier run was
# started outside SLURM, was killed with the shell that owned it, and stranded 40,600 of
# 113,682 proteins; running it under sbatch is what stops that recurring.
#
# It also resubmits itself when it runs out of wall clock with the crawl unfinished, which
# is the case this job is most likely to hit - the runtime depends on how hard InterPro is
# rate-limiting on the day, and that is not knowable at submit time.
#
# Submit:
#   sbatch scripts/slurm/fetch_hsp70_rockfish.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/reu_domain_layout_v2}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"
# Seconds before a single request is abandoned. The crawl paces itself internally; what
# it needs from us is patience with a slow page, not a rate cap.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
MAX_RESUBMITS="${MAX_RESUBMITS:-6}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"

cd "${PROJECT_DIR}"

# shellcheck source=scripts/slurm/rockfish_common.sh
source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

ml anaconda3/2024.02-1
rockfish_activate_env "${CONDA_ENV}"

if ! command -v fetch-proteins-dnak > /dev/null 2>&1; then
	printf "fetch-proteins-dnak not found after activating conda env '%s'\n" "${CONDA_ENV}" 1>&2
	printf "From the repo root: pip install -e \".[structure]\"\n" 1>&2
	exit 1
fi

CHECKPOINT="${PROJECT_DIR}/ipr013126_checkpoint.json"
OUTPUT="${PROJECT_DIR}/ipr013126_proteins.json"

# Report where the crawl is resuming from, so the log says what was inherited rather than
# leaving a reader to infer it from the runtime.
if [[ -f ${CHECKPOINT} ]]; then
	python3 - "${CHECKPOINT}" << 'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)
fetched = len(state.get("proteins", ()))
total = state.get("proteins_reported", 0)
print(f"resuming: {fetched} of {total} proteins already fetched", flush=True)
PY
else
	printf "no checkpoint; starting a fresh crawl\n"
fi

set +e
fetch-proteins-dnak --output "${OUTPUT}" --checkpoint "${CHECKPOINT}" --timeout "${REQUEST_TIMEOUT}"
STATUS=$?
set -e

if [[ ${STATUS} -eq 0 ]]; then
	printf "fetch complete: %s\n" "${OUTPUT}"
	exit 0
fi

# Unfinished. Resubmit from the checkpoint unless we have already done so too often, which
# would mean the crawl is stuck rather than slow and a human should look at it.
if [[ ${RESUBMIT_COUNT} -ge ${MAX_RESUBMITS} ]]; then
	printf "fetch still incomplete after %s resubmits; not resubmitting again\n" "${RESUBMIT_COUNT}" 1>&2
	exit 1
fi

NEXT=$((RESUBMIT_COUNT + 1))
printf "fetch incomplete (exit %s); resubmitting (%s of %s)\n" "${STATUS}" "${NEXT}" "${MAX_RESUBMITS}"
sbatch --export=ALL,RESUBMIT_COUNT="${NEXT}" "${PROJECT_DIR}/scripts/slurm/fetch_hsp70_rockfish.sh"
