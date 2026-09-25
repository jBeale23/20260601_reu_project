#!/bin/bash -ue
#SBATCH --job-name=string-partners
#SBATCH --partition=shared
#SBATCH --time=00-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 217,701 requests against a public service, at modest concurrency because STRING asks
# callers to identify themselves and rate-limits accordingly. Checkpointed every 2,000.
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${HOME}/layout_venv"

cd "${PROJ}"
"${VENV}/bin/python" scripts/cluster/string_run.py
