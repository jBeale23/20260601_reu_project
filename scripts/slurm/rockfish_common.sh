#!/bin/bash
# Shared helpers for Rockfish SLURM array jobs (affetch and pocket charge).
# Source from worker scripts: source "${PROJECT_DIR}/scripts/slurm/rockfish_common.sh"

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
	if ! grep -qxF "${accession}" "${completion_log}" 2> /dev/null; then
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

# True when accession already listed in completion log.
rockfish_is_completed() {
	local accession="$1"
	local completion_log="$2"
	grep -qxF "${accession}" "${completion_log}" 2> /dev/null
}

# Append accession to a failed log once (atomic).
rockfish_log_failure() {
	local accession="$1"
	local failed_log="$2"
	local lock_file="$3"

	[[ -n ${accession} ]] || return 0

	exec 8>> "${lock_file}"
	flock -x 8
	if ! grep -qxF "${accession}" "${failed_log}" 2> /dev/null; then
		printf "%s\n" "${accession}" >> "${failed_log}"
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
