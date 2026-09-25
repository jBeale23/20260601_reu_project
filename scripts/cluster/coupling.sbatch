#!/bin/bash -ue
#SBATCH --job-name=coupling
#SBATCH --partition=parallel
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Sequence-structure-function coupling, over the AlphaFold models.
#
# Which of the three has moved: pairs where sequence diverged but the fold held, where both
# held, and - rarest - where the fold moved while the sequence did not. Foldseek rather than
# Dali: Dali is the more established method and is what a reviewer asks about, but an
# all-versus-all over 142,948 models is far outside what it can do, and Foldseek reaches the
# same fold-level conclusions through a structural alphabet.
WK="${HOME}/scr4_sfried3/domain_layout_run"
# Results go to ANALYSIS_OUT_DIR when it is set. The scr4_sfried3 group quota is shared with
# the rest of the lab and fills without warning; when it is full every write here fails while
# reads keep working, so a run can finish its computation and have nowhere to put the answer.
# Inputs still come from WK, so redirecting output moves nobody else's data.
OUT="${ANALYSIS_OUT_DIR:-${WK}}"
mkdir -p "${OUT}"
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"
STRUCTURES="${STRUCTURES:-${HOME}/scr4_sfried3/alphafoldfetch/structures}"
# AlphaFold models arrive gzipped and Foldseek's directory scan skips .gz without error,
# building an empty database - which is how a two-hour all-versus-all returned zero hits.
# One format is staged, not both, or every protein self-hits against its own twin.
# Stage to node-local NVMe, not to shared scratch. Compute nodes mount /tmp on a 1.7 TB
# local partition; the group quota on /scratch4 is a few gigabytes and staging into it
# exhausted the quota three times, each time killing unrelated jobs mid-write. Node-local
# storage is the right place for scratch that exists only for the life of the job, and it
# makes the species and random strategies - 15,188 and 142,948 models - possible at all.
STAGE_ROOT="${STAGE_ROOT:-/tmp/${SLURM_JOB_ID:-$$}}"
STAGE_DIR="${STAGE_DIR:-${STAGE_ROOT}/staged_${STRATEGY:-architecture}}"
mkdir -p "${STAGE_DIR}"
# Node-local storage is not cleaned between jobs on every scheduler, so remove it here
# rather than leaving tens of gigabytes behind for whoever lands on this node next.
trap 'rm -rf "${STAGE_ROOT}"' EXIT

# Which proteins enter the all-versus-all. Every model is ~10^10 pairs and, staged, exceeds
# the shared quota - it also weights the answer toward whichever families are over-sequenced.
# "architecture" is one protein per domain architecture, this project's own unit. Re-run with
# STRATEGY=species / genus / random to check the result does not depend on the choice.
STRATEGY="${STRATEGY:-architecture}"
FOLDSEEK="${PROJ}/vendor/foldseek/bin/foldseek"

[[ -x ${FOLDSEEK} ]] || {
	printf "foldseek not found at %s\n" "${FOLDSEEK}" 1>&2
	exit 1
}
[[ -d ${STRUCTURES} ]] || {
	printf "structure directory not found: %s\n" "${STRUCTURES}" 1>&2
	exit 1
}

# The function store is written by the transfer job and may not exist yet; the analysis
# runs without it, omitting the shared-function column rather than inventing one.
FUNCTIONS_ARG=()
[[ -f ${WK}/function_store.json ]] && FUNCTIONS_ARG=(--functions "${WK}/function_store.json")

cd "${PROJ}"
printf "[%s] coupling analysis, %s threads\n" "$(date +%T)" "${SLURM_CPUS_PER_TASK}"
"${VENV}/bin/analyze-coupling" "${STRUCTURES}" \
	--domain-store "${WK}/protein_domains_all_architectures.json" \
	--strategy "${STRATEGY}" \
	--structure-features "${WK}/structure_features_surface.csv" \
	--stage-dir "${STAGE_DIR}" \
	--stage-suffix .pdb \
	--foldseek "${FOLDSEEK}" \
	--threads "${SLURM_CPUS_PER_TASK}" \
	"${FUNCTIONS_ARG[@]}" \
	--max-structures "${MAX_STRUCTURES:-200000}" \
	-o "${OUT}/coupling_report_${STRATEGY}.json"
printf "[%s] done\n" "$(date +%T)"
