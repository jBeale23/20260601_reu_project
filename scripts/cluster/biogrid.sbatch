#!/bin/bash -ue
#SBATCH --job-name=biogrid
#SBATCH --partition=shared
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Fetch BioGRID interaction evidence for the J-domain proteins.
#
# Without this, four of the seven evidence combinations in the pairing sweep - every one
# involving interactions - have nothing to test, and the sweep reduces to co-occurrence
# alone. The access key is read from ~/.biogrid_key (mode 600, outside any git tree) and is
# never passed as an argument, so it cannot appear in a process listing or a job script.
WK="${HOME}/scr4_sfried3/domain_layout_run"
# Results go to ANALYSIS_OUT_DIR when it is set. The scr4_sfried3 group quota is shared with
# the rest of the lab and fills without warning; when it is full every write here fails while
# reads keep working, so a run can finish its computation and have nowhere to put the answer.
# Inputs still come from WK, so redirecting output moves nobody else's data.
OUT="${ANALYSIS_OUT_DIR:-${WK}}"
mkdir -p "${OUT}"
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

cd "${PROJ}"
printf "[%s] fetching BioGRID interactions\n" "$(date +%T)"
"${VENV}/bin/fetch-biogrid-interactions" "${WK}/ipr001623_all_architectures.json" \
	-o "${OUT}/biogrid_interactions.json" \
	--key-file "${HOME}/.biogrid_key"
printf "[%s] done\n" "$(date +%T)"
