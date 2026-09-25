#!/bin/bash -ue
#SBATCH --job-name=function-transfer
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Ask for a signal five minutes before the wall clock, delivered to the batch shell.
# Without this the time-limit kill terminates the whole job step, so the resubmit branch
# below never runs - which is how a fetch died at 176,000 of 181,526 with nobody notified.
#SBATCH --signal=B:USR1@300

# Fetch functional annotation, transfer it by homology, and test whether the structural
# split predicts function.
#
# The measurement that makes transfer necessary: of 400 proteins sampled, 74% carry Gene
# Ontology terms but only 0.8% are reviewed, and an unreviewed entry carried zero evidence
# records. Those terms are electronically inferred, most often by InterPro2GO, where the
# domain signature implies the term - so testing whether architecture predicts function
# using them would assume the answer. Only experimentally backed annotation can donate.
WK="${HOME}/scr4_sfried3/domain_layout_run"
PROJ="${HOME}/repositories/reu_domain_layout_v2"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="${PROJ}/vendor/foldseek/bin:${PATH}"

cd "${PROJ}"
# The fetch runs at roughly 1.7 accessions/s against a rate-limited UniProt, so 181,526
# needs about thirty hours - longer than any single allocation here. It checkpoints every
# 2,000, so the work survives a kill; this resubmits from that checkpoint rather than
# needing someone to notice and do it by hand.
MAX_RESUBMITS="${MAX_RESUBMITS:-4}"

# Resubmit on the wall-clock signal too, not only on a nonzero exit.
resubmit_on_signal() {
	if [[ ${RESUBMIT_COUNT:-0} -lt ${MAX_RESUBMITS} ]]; then
		printf "[%s] wall clock reached; resubmitting from the checkpoint\n" "$(date +%T)"
		cd "${WK}"
		sbatch --export=ALL,RESUBMIT_COUNT="$((${RESUBMIT_COUNT:-0} + 1))" transfer.sbatch "$@"
	fi
	exit 0
}
trap 'resubmit_on_signal "$@"' USR1
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"

printf "[%s] functional annotation and homology transfer (attempt %s)\n" "$(date +%T)" "$((RESUBMIT_COUNT + 1))"
set +e
"${HOME}/layout_venv/bin/python" "${WK}/run_transfer.py" "$@"
STATUS=$?
set -e

if [[ ${STATUS} -eq 0 ]]; then
	printf "[%s] done\n" "$(date +%T)"
	exit 0
fi

if [[ ${RESUBMIT_COUNT} -ge ${MAX_RESUBMITS} ]]; then
	printf "[%s] still incomplete after %s resubmits; stopping\n" "$(date +%T)" "${RESUBMIT_COUNT}" 1>&2
	exit 1
fi

NEXT=$((RESUBMIT_COUNT + 1))
printf "[%s] incomplete (exit %s); resubmitting %s of %s\n" "$(date +%T)" "${STATUS}" "${NEXT}" "${MAX_RESUBMITS}"
cd "${WK}"
sbatch --export=ALL,RESUBMIT_COUNT="${NEXT}" transfer.sbatch "$@"
