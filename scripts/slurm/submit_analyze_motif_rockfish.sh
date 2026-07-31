#!/bin/bash -ue

# Submit DnaJ motif / charge-window conservation on Rockfish.
#
# Sequence-based (no PDB required). Uses the pocket conda env by default.
#
# Usage:
#   export FETCH_JSON="${WK_DIR}/ipr001623_domain_architectures_no_dedup.json"
#   bash scripts/slurm/submit_analyze_motif_rockfish.sh
#
# Optional held-out DnaK sanity check after pocket results exist:
#   HELD_OUT_POCKET_CSV="${WK_DIR}/pocket_results/pocket_charge_summary.csv" \
#     bash scripts/slurm/submit_analyze_motif_rockfish.sh

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/motif_results}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/analyze_motif_rockfish.sh"
FETCH_JSON="${FETCH_JSON:-${WK_DIR}/ipr001623_domain_architectures_no_dedup.json}"
CONDA_ENV="${CONDA_ENV:-${HOME}/pocket}"

[[ -f ${JOB_SCRIPT} ]] || {
	printf "SLURM job script not found: %s\n" "${JOB_SCRIPT}" 1>&2
	exit 1
}

[[ -f ${FETCH_JSON} ]] || {
	printf "DnaJ fetch JSON not found: %s\n" "${FETCH_JSON}" 1>&2
	printf "Copy the architecture JSON to WK_DIR or set FETCH_JSON.\n" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${WK_DIR}/logs"

printf "Submitting motif-conservation job for %s\n" "${FETCH_JSON}"
printf "Results directory: %s\n" "${RESULTS_DIR}"

sbatch_args=(
	--output="${WK_DIR}/logs/motif_%j.out"
	--error="${WK_DIR}/logs/motif_%j.err"
)
sbatch_export="ALL,PROJECT_DIR=${PROJECT_DIR},WK_DIR=${WK_DIR},RESULTS_DIR=${RESULTS_DIR},FETCH_JSON=${FETCH_JSON},CONDA_ENV=${CONDA_ENV}"
if [[ -n ${HELD_OUT_POCKET_CSV:-} ]]; then
	sbatch_export+=",HELD_OUT_POCKET_CSV=${HELD_OUT_POCKET_CSV}"
fi
if [[ -n ${MAX_PER_FAMILY:-} ]]; then
	sbatch_export+=",MAX_PER_FAMILY=${MAX_PER_FAMILY}"
fi
if [[ -n ${MIN_MEMBERS:-} ]]; then
	sbatch_export+=",MIN_MEMBERS=${MIN_MEMBERS}"
fi
sbatch_args+=(--export="${sbatch_export}")

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Track logs under %s/logs/motif_%s.{out,err}\n" "${WK_DIR}" "${job_id}"
printf "\nAfter completion, join features:\n"
printf "  merge-all-features --dnaj-json \"%s\" \\\n" "${FETCH_JSON}"
printf "    --motif-csv \"%s/motif_accession_features.csv\" \\\n" "${RESULTS_DIR}"
printf "    -o \"%s/merged_features/dnaj_with_motif.csv\"\n" "${WK_DIR}"
