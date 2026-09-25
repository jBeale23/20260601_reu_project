#!/bin/bash -ue
#SBATCH --job-name=validate-v6
#SBATCH --partition=parallel
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Validation over the store that actually contains all 4,155 domain architectures.
#
# v5 and everything before it ran on 181,526 proteins carrying exactly 20 distinct
# architecture ids - the top-20 architectures. The other 4,135 architectures had been
# fetched (36,175 proteins, 91 chunks) but never merged, so every result to date describes
# the common architectures only. This runs the same analysis over the merged 217,701-protein
# store; the pair of reports is what says what the rare architectures changed.
#
# Longer wall clock than v5: 20% more proteins, and the rare architectures are the ones the
# layout stage finds hardest, being the least like anything the rules were written against.
#
# Every previous run was grammar-only. The structural features were computed - 142,948
# models, 78.75% coverage - and written to CSV, but no command-line option could hand them
# to the report, so `structural_features` was always empty and both structural variants
# were silently skipped. --structural-features is that option.
#
# This run therefore adds, over v4:
#   grammar_structure_excluded  - proteins without a model dropped
#   grammar_structure_imputed   - missing features median-filled, with a has_structure flag
#   redundancy_clustering       - 30% identity, on full / J-domain / non-J regions
#   chaperone_evidence          - whether rare architectures carry the HPD motif at all
WK="${HOME}/scr4_sfried3/domain_layout_run"
# Results go to ANALYSIS_OUT_DIR when it is set. The scr4_sfried3 group quota is shared with
# the rest of the lab and fills without warning; when it is full every write here fails while
# reads keep working, so a run can finish its computation and have nowhere to put the answer.
# Inputs still come from WK, so redirecting output moves nobody else's data.
OUT="${ANALYSIS_OUT_DIR:-${WK}}"
mkdir -p "${OUT}"
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# shellcheck source=/dev/null
source "${PROJ}/scripts/slurm/rockfish_common.sh"
rockfish_ensure_mafft "${PROJ}/vendor/mafft" "" || true

# The surface table is a superset of the base one - same rows, plus SASA and pocket
# geometry - so it is preferred when present.
FEATURES="${WK}/structure_features_surface.csv"
[[ -f ${FEATURES} ]] || FEATURES="${WK}/structure_features.csv"
[[ -f ${FEATURES} ]] || {
	printf "no structural feature table found in %s\n" "${WK}" 1>&2
	exit 1
}

cd "${PROJ}"
printf "[%s] validation v5, structural features from %s\n" "$(date +%T)" "${FEATURES}"
"${VENV}/bin/python" -m validation.cli "${WK}/protein_domains_all_architectures.json" \
	-o "${OUT}/validation_report_v6_all_architectures.json" \
	--structural-features "${FEATURES}" \
	--permutations 200 --recurrence-permutations 1000 --seed 0 \
	--workers "${SLURM_CPUS_PER_TASK}"
printf "[%s] done\n" "$(date +%T)"
