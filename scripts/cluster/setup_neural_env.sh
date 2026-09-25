#!/bin/bash -ue
# Build the GPU environment for the neural challenger.
#
# Separate from layout_venv on purpose: that environment holds a CPU-only torch, and the
# CUDA build is several gigabytes. Overwriting it would make every existing analysis job
# depend on a GPU stack none of them use.
VENV="${NEURAL_VENV:-${HOME}/neural_venv}"
PROJ="${PROJ:-${HOME}/repositories/reu_domain_layout_v2}"

ml anaconda3/2024.02-1 2> /dev/null || true

# The login node's default python3 is 3.6, which no current torch wheel supports and which
# produced a venv whose pip could not even install setuptools. The conda environment behind
# layout_venv carries 3.12, so use that interpreter explicitly rather than whatever
# `python3` happens to resolve to.
BASE_PYTHON="${BASE_PYTHON:-${HOME}/pocket/bin/python3.12}"
if [[ ! -x ${BASE_PYTHON} ]]; then
	BASE_PYTHON="$(command -v python3.12 || command -v python3.11 || true)"
fi
if [[ -z ${BASE_PYTHON} || ! -x ${BASE_PYTHON} ]]; then
	printf "no python 3.11+ found; set BASE_PYTHON to one\n" 1>&2
	exit 1
fi
printf "building %s with %s (%s)\n" "${VENV}" "${BASE_PYTHON}" "$("${BASE_PYTHON}" --version)"

if [[ ! -d ${VENV} ]]; then
	"${BASE_PYTHON}" -m venv "${VENV}"
fi

"${VENV}/bin/pip" install --upgrade pip -q
# cu121 matches the driver on the a100 partition; pinning the index avoids pulling the
# CPU-only wheel that is the default on PyPI.
"${VENV}/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121
"${VENV}/bin/pip" install -q transformers peft
"${VENV}/bin/pip" install -q -e "${PROJ}"

"${VENV}/bin/python" - << 'PY'
import torch

print("torch", torch.__version__, "| cuda build:", torch.version.cuda, "| available:", torch.cuda.is_available())
PY
