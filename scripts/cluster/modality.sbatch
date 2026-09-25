#!/bin/bash -ue
#SBATCH --job-name=modality-info
#SBATCH --partition=shared
#SBATCH --time=00-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Whether grammar and structure carry the same information about function, or different
# information. Memory is generous because the transferred-annotation store is 243 MB of JSON,
# but capped at 96G: on the shared partition a larger request implies more CPUs than the
# QOS allows and the job sits in QOSMaxCpuPerJobLimit forever rather than failing.
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

cd "${PROJ}"
"${VENV}/bin/python" scripts/cluster/modality_run.py
