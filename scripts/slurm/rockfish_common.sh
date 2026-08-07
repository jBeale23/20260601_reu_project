#!/bin/bash
# Shared helpers for Rockfish SLURM array jobs (affetch, pocket charge, domain fetch).
# Source from worker scripts: source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

# Activate a Python environment that may be either a conda env or a plain venv.
#
# `conda activate` fails on a venv ("Not a conda environment") and `source bin/activate`
# is wrong for a conda env, so the worker cannot assume one flavour. A venv is identified
# by its pyvenv.cfg marker.
rockfish_activate_env() {
	local env_path="$1"

	if [[ -f ${env_path}/pyvenv.cfg ]]; then
		# shellcheck disable=SC1091  # path is only known at run time
		source "${env_path}/bin/activate"
	else
		conda activate "${env_path}"
	fi
}

# First 1-based snapshot line owned by a chunked array task.
rockfish_chunk_start() {
	local task_id="$1"
	local chunk_size="$2"

	printf "%s\n" "$(((task_id - 1) * chunk_size + 1))"
}

# Per-chunk marker used in completion/failure logs and output filenames.
#
# The marker combines the starting line with a digest of the accessions in the chunk, so
# it identifies *content*, not just a position. If the master accession list is ever
# rebuilt and line offsets shift, the digest changes and the chunk is re-fetched instead
# of being silently skipped as "already done" while covering different proteins.
rockfish_chunk_marker() {
	local start="$1"
	local queue_file="${2:-}"
	local chunk_size="${3:-1}"
	local digest="nocontent"

	if [[ -n ${queue_file} && -f ${queue_file} ]]; then
		digest="$(sed -n "${start},$((start + chunk_size - 1))p" "${queue_file}" | sha1sum | cut -c1-8)"
	fi

	printf "chunk_%06d_%s\n" "${start}" "${digest}"
}

# Read accession for this array task from a submit-time snapshot (fixed line mapping).
rockfish_read_task_accession() {
	local task_id="$1"
	local queue_file="$2"

	if [[ -z ${queue_file} || ! -f ${queue_file} ]]; then
		printf "ARRAY_QUEUE_FILE is missing or not found: '%s'\n" "${queue_file}" 1>&2
		printf "Re-submit with scripts/slurm/submit_*_rockfish.sh (writes a snapshot at submit time).\n" 1>&2
		return 1
	fi

	sed -n "${task_id}p" "${queue_file}"
}

# Append accession to a completion log once (atomic, no duplicate lines).
rockfish_mark_completed() {
	local accession="$1"
	local completion_log="$2"
	local lock_file="$3"

	[[ -n ${accession} ]] || return 0

	exec 9>> "${lock_file}"
	flock -x 9
	if ! awk -F '\t' -v acc="${accession}" '$1 == acc { found = 1 } END { exit !found }' "${completion_log}" 2> /dev/null; then
		printf "%s\n" "${accession}" >> "${completion_log}"
	fi
}

# True when an AlphaFold structure for this accession already exists on disk.
rockfish_structure_exists() {
	local accession="$1"
	local structures_dir="$2"
	local model_version="$3"
	local base="${structures_dir}/AF-${accession}-F1-model_v${model_version}"

	[[ -f ${base}.pdb.gz || -f ${base}.pdb || -f ${base}.cif.gz || -f ${base}.cif ]]
}

# True when accession already listed in completion log (first column).
rockfish_is_completed() {
	local accession="$1"
	local completion_log="$2"
	awk -F '\t' -v acc="${accession}" '$1 == acc { found = 1 } END { exit !found }' "${completion_log}" 2> /dev/null
}

# Append accession (+ optional reason) to a failed log once (atomic).
# Format: accession<TAB>reason_code<TAB>detail
rockfish_log_failure() {
	local accession="$1"
	local failed_log="$2"
	local lock_file="$3"
	local reason_code="${4:-unknown}"
	local detail="${5:-}"

	[[ -n ${accession} ]] || return 0

	exec 8>> "${lock_file}"
	flock -x 8
	if ! awk -F '\t' -v acc="${accession}" '$1 == acc { found = 1 } END { exit !found }' "${failed_log}" 2> /dev/null; then
		printf "%s\t%s\t%s\n" "${accession}" "${reason_code}" "${detail}" >> "${failed_log}"
	fi
}

# Resolve AF-<accession>-F1-model_v<version> PDB path for analyze-pocket-charge (.pdb.gz or .pdb).
rockfish_resolve_pdb_structure() {
	local accession="$1"
	local structures_dir="$2"
	local model_version="$3"
	local base="${structures_dir}/AF-${accession}-F1-model_v${model_version}"

	if [[ -f ${base}.pdb.gz ]]; then
		printf "%s\n" "${base}.pdb.gz"
	elif [[ -f ${base}.pdb ]]; then
		printf "%s\n" "${base}.pdb"
	fi
}

# Resolve AF-<accession>-F1-model_v<version> structure path (any affetch format).
rockfish_resolve_structure() {
	local accession="$1"
	local structures_dir="$2"
	local model_version="$3"
	local base="${structures_dir}/AF-${accession}-F1-model_v${model_version}"

	if [[ -f ${base}.pdb.gz ]]; then
		printf "%s\n" "${base}.pdb.gz"
	elif [[ -f ${base}.pdb ]]; then
		printf "%s\n" "${base}.pdb"
	elif [[ -f ${base}.cif.gz ]]; then
		printf "%s\n" "${base}.cif.gz"
	elif [[ -f ${base}.cif ]]; then
		printf "%s\n" "${base}.cif"
	fi
}

# Classify why a PDB is missing for pocket analysis. Prints reason_code to stdout.
# Exports ROCKFISH_MISSING_DETAIL with a human-readable message for the caller.
rockfish_diagnose_missing_pdb() {
	local accession="$1"
	local structures_dir="$2"
	local model_version="$3"
	local base="${structures_dir}/AF-${accession}-F1-model_v${model_version}"

	if [[ -f ${base}.cif.gz || -f ${base}.cif ]]; then
		export ROCKFISH_MISSING_DETAIL="CIF present but PDB required for pocket charge; re-fetch with FILE_TYPE including p (e.g. pz or pcz)"
		printf "cif_only\n"
		return 0
	fi

	if [[ -f ${base}.pdb.gz || -f ${base}.pdb ]]; then
		export ROCKFISH_MISSING_DETAIL="PDB path unexpectedly unresolved for ${base}"
		printf "missing_pdb\n"
		return 0
	fi

	export ROCKFISH_MISSING_DETAIL="No AF PDB/CIF under ${structures_dir} for model v${model_version}"
	printf "missing_any_structure\n"
}
