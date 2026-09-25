#!/bin/bash -ue

# Submit the dual-layer domain-layout analysis on Rockfish.
#
# Prerequisites:
#   bash scripts/slurm/submit_fetch_domains_rockfish.sh      # domain stores per chunk
#   fetch-protein-domains --merge-stores "${WK_DIR}/domain_results" \
#     -o "${WK_DIR}/protein_domains.json"
#
# Usage:
#   bash scripts/slurm/submit_analyze_domain_layout_rockfish.sh
#
# Set DISORDER_BACKEND=foldindex / SHARK_BACKEND=blosum_kmer to force the built-in
# fallbacks when metapredict / bio-shark are not installed in the conda env.

WK_DIR="${WK_DIR:-${HOME}/scr4_sfried3/alphafoldfetch}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/repositories/20260601_reu_project}"
RESULTS_DIR="${RESULTS_DIR:-${WK_DIR}/layout_results}"
DOMAIN_STORE="${DOMAIN_STORE:-${WK_DIR}/protein_domains.json}"
JOB_SCRIPT="${PROJECT_DIR}/scripts/slurm/analyze_domain_layout_rockfish.sh"
CONDA_ENV="${CONDA_ENV:-${HOME}/layout}"
DISORDER_BACKEND="${DISORDER_BACKEND:-auto}"
SHARK_BACKEND="${SHARK_BACKEND:-auto}"
# Disorder prediction is CPU-bound and scales close to linearly with workers, so the
# resource request is overridable: the whole DnaJ set is ~17 h on 8 cores but ~6 h on 24.
CPUS_PER_TASK="${CPUS_PER_TASK:-}"
JOB_MEMORY="${JOB_MEMORY:-}"

[[ -f ${JOB_SCRIPT} ]] || {
	printf "SLURM job script not found: %s\n" "${JOB_SCRIPT}" 1>&2
	exit 1
}

[[ -f ${DOMAIN_STORE} ]] || {
	printf "Domain store not found: %s\n" "${DOMAIN_STORE}" 1>&2
	printf "Merge the per-chunk stores first:\n" 1>&2
	printf "  fetch-protein-domains --merge-stores \"%s/domain_results\" -o \"%s\"\n" "${WK_DIR}" "${DOMAIN_STORE}" 1>&2
	exit 1
}

mkdir -p "${WK_DIR}" "${RESULTS_DIR}" "${WK_DIR}/logs"

printf "Submitting domain-layout job for %s\n" "${DOMAIN_STORE}"
printf "Results directory: %s\n" "${RESULTS_DIR}"

sbatch_args=(
	--output="${WK_DIR}/logs/layout_%j.out"
	--error="${WK_DIR}/logs/layout_%j.err"
)
if [[ -n ${CPUS_PER_TASK} ]]; then
	sbatch_args+=(--cpus-per-task="${CPUS_PER_TASK}")
fi
if [[ -n ${JOB_MEMORY} ]]; then
	sbatch_args+=(--mem="${JOB_MEMORY}")
fi
sbatch_export="ALL,PROJECT_DIR=${PROJECT_DIR},WK_DIR=${WK_DIR},RESULTS_DIR=${RESULTS_DIR}"
sbatch_export+=",DOMAIN_STORE=${DOMAIN_STORE},CONDA_ENV=${CONDA_ENV}"
sbatch_export+=",DISORDER_BACKEND=${DISORDER_BACKEND},SHARK_BACKEND=${SHARK_BACKEND}"
if [[ -n ${WORKERS:-} ]]; then
	sbatch_export+=",WORKERS=${WORKERS}"
fi
if [[ -n ${MAX_PROTEINS:-} ]]; then
	sbatch_export+=",MAX_PROTEINS=${MAX_PROTEINS}"
fi
sbatch_args+=(--export="${sbatch_export}")

if [[ -n ${SLURM_ACCOUNT:-} ]]; then
	sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

submit_output="$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")"
job_id="${submit_output##* }"
printf "%s\n" "${submit_output}"
printf "Track logs under %s/logs/layout_%s.{out,err}\n" "${WK_DIR}" "${job_id}"
printf "\nAfter completion, join every feature table:\n"
printf "  merge-all-features --dnaj-json ipr001623_domain_architectures_no_dedup.json \\\n"
printf "    --domain-csv \"%s/domain_layout_features.csv\" \\\n" "${RESULTS_DIR}"
printf "    -o \"%s/merged_features/all_features.csv\"\n" "${WK_DIR}"
