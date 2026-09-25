#!/bin/bash -ue
#SBATCH --job-name=partner-predict
#SBATCH --partition=shared
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

cd "${PROJ}"
"${VENV}/bin/python" scripts/cluster/partner_prediction_run.py
