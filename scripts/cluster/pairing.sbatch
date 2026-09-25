#!/bin/bash -ue
#SBATCH --job-name=pairing-sweep
#SBATCH --partition=shared
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# The full JDP-Hsp70 pairing sweep: every combination of granularity and evidence.
#
# Fifteen granularity combinations - each non-empty subset of {compartment, functional
# subtype, paralogue, partner scope} - times seven evidence combinations - each non-empty
# subset of {BioGRID interactions, genus co-occurrence, species co-occurrence}. One hundred
# and five tests, Benjamini-Hochberg corrected together, because reporting the best of a
# hundred uncorrected would manufacture a result.
#
# BioGRID is optional here: if the file is absent the interaction combinations run with an
# empty partner set and are reported as untested rather than as null results.
WK="${HOME}/scr4_sfried3/domain_layout_run"
# Results go to ANALYSIS_OUT_DIR when it is set. The scr4_sfried3 group quota is shared with
# the rest of the lab and fills without warning; when it is full every write here fails while
# reads keep working, so a run can finish its computation and have nowhere to put the answer.
# Inputs still come from WK, so redirecting output moves nobody else's data.
OUT="${ANALYSIS_OUT_DIR:-${WK}}"
mkdir -p "${OUT}"
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

BIOGRID="${WK}/biogrid_interactions_by_accession.json"
BIOGRID_ARG=()
if [[ -f ${BIOGRID} ]]; then
	BIOGRID_ARG=(--biogrid "${BIOGRID}")
else
	printf "no BioGRID file at %s; interaction combinations will be empty\n" "${BIOGRID}" 1>&2
fi

cd "${PROJ}"
printf "[%s] pairing sweep\n" "$(date +%T)"
"${VENV}/bin/analyze-jdp-hsp70-pairing" "${WK}/protein_domains_backfilled.json" \
	--layout-features "${WK}/full_layout_v4/domain_layout_features.csv" \
	--hsp70 "${PROJ}/ipr013126_proteins.json" \
	"${BIOGRID_ARG[@]}" \
	-o "${OUT}/pairing_sweep.json"
printf "[%s] done\n" "$(date +%T)"
