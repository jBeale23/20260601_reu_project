#!/bin/bash -ue
#SBATCH --job-name=neural-challenger
#SBATCH --partition=a100
# GPU work bills a different account from CPU work: sfried3 carries only the normal
# QOS, which the a100 partition rejects, while sfried3_gpu carries qos_gpu. Setting
# one without the other leaves the job queued forever or refused at submit.
#SBATCH --account=sfried3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# The hierarchical neural challenger: grammar CNN/BiLSTM, syntax transformer, dual heads.
#
# Read the result against the label set, not the architecture. There are 129 labelled
# proteins here, an effective count nearer 39 once cross-genus ortholog groups are
# collapsed, and roughly 150,000 trainable parameters - over a thousand per example. That
# ratio, not the model, is what limits this comparison, and validation.capacity quantifies
# how fine a distinction 129 labels can certify at all.
#
# Folds are clustered at 30% identity on the non-J regions rather than drawn at random. A
# model this size memorises, and random folds leave near-identical relatives of every
# held-out protein in training, which a memorising model scores well on.
#
# ESM-2 trains in the loop rather than supplying precomputed embeddings: frozen embeddings
# would leave the LoRA adapters without a gradient, making the "+ LoRA" half of the design
# decorative. The encoder is rebuilt per fold, so adapters fitted on one fold's training set
# never see the next fold's held-out proteins.
WK="${HOME}/scr4_sfried3/domain_layout_run"
# Results go to ANALYSIS_OUT_DIR when it is set. The scr4_sfried3 group quota is shared with
# the rest of the lab and fills without warning; when it is full every write here fails while
# reads keep working, so a run can finish its computation and have nowhere to put the answer.
# Inputs still come from WK, so redirecting output moves nobody else's data.
OUT="${ANALYSIS_OUT_DIR:-${WK}}"
mkdir -p "${OUT}"
PROJ="${HOME}/repositories/reu_domain_layout_v2"
VENV="${NEURAL_VENV:-${HOME}/neural_venv}"

[[ -x ${VENV}/bin/train-neural-challenger ]] || {
	printf "neural environment not found at %s\n" "${VENV}" 1>&2
	printf "Create it with scripts/cluster/setup_neural_env.sh\n" 1>&2
	exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# ESM-2 weights are pre-cached on the login node and the run is pinned offline: a compute
# node reaching for the Hub mid-run would either stall on a rate limit or fail outright,
# and the failure would land after the layout stage had already been paid for.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

[[ -d ${HF_HOME} ]] || {
	printf "ESM-2 cache not found at %s; pre-download it on a login node first\n" "${HF_HOME}" 1>&2
	exit 1
}

cd "${PROJ}"
printf "[%s] neural challenger\n" "$(date +%T)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || printf "no GPU visible\n"

"${VENV}/bin/train-neural-challenger" "${WK}/protein_domains_backfilled.json" \
	-o "${OUT}/neural_report.json" \
	--epochs 60 --batch-size 8 --folds 5 --seeds 0 1 2 3 4 \
	--workers "${SLURM_CPUS_PER_TASK}"
printf "[%s] done\n" "$(date +%T)"
