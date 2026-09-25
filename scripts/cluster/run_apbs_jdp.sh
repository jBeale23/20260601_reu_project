#!/bin/bash -u
# Deliberately not -e. Every structure in this loop can fail for benign reasons - a missing
# model, a conversion that rejects an odd residue, an APBS run that does not converge - and
# under -e the first such failure killed the whole array task. That is how a 4,396-target
# run produced 14 results: each task processed a couple of structures and died silently on
# the third. Failures are handled per structure instead, so one bad model costs one model.
#SBATCH --job-name=apbs-jdp
#SBATCH --partition=shared
#SBATCH --time=00-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --array=1-110%40
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Poisson-Boltzmann electrostatics for J-domain proteins.
#
# Adapted from another project's APBS pipeline in rockfish-projects/apbs, which is left
# untouched; the converter and scraper were copied here and the numerical settings kept
# identical so the two sets of energies are directly comparable.
#
# Two differences from the original. Its inputs are uncompressed .cif; AlphaFold models here
# arrive gzipped, so each is decompressed into node-local scratch first. And it runs one
# structure per array task; a whole-proteome run at that granularity would be 142,948 tasks,
# so each task here takes a contiguous chunk.
#
# The energies are what the formal-charge sums in structure_features cannot give: a real
# solvation free energy, which is what an affinity calculation needs.
# The group quota on scr4_sfried3 is shared with the rest of the lab and fills without
# warning; when it does, writes fail and array tasks die before they can even open a log.
# WK is overridable so the run can be moved to a filesystem that still has room without
# touching anyone else's data.
WK="${APBS_WORK_DIR:-${HOME}/scr4_sfried3/domain_layout_run/apbs}"
CODE="${WK}/code"
SIF="${HOME}/repositories/rockfish-projects/apbs/env_files/apbs.sif"
STRUCTURES="${HOME}/scr4_sfried3/alphafoldfetch/structures"
INPUT_FILE="${WK}/targets.txt"
COMPLETION_LOG="${WK}/completed.txt"
OUTPUT_DIR="${WK}/outputs"
CHUNK="${CHUNK:-40}"

module load anaconda3/2024.02-1
# Singularity refuses to load on a login node, so this only resolves inside the job.
module load singularity/3.8.7
conda activate "${HOME}/apbsenv"

SCRATCH="/tmp/apbs_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${SCRATCH}" "${OUTPUT_DIR}"
trap 'rm -rf "${SCRATCH}"' EXIT

# This filesystem cannot mount a SIF directly, so singularity unpacks the image into a
# temporary sandbox first. It puts that sandbox on scratch4 by default, which is exactly the
# filesystem whose group quota keeps filling; when it is full the unpack fails and every
# APBS call in the task dies. Because the call sends stderr to /dev/null, the task then exits
# zero having produced nothing - which is what a 4,396-target run reporting 14 results looks
# like. Pointing singularity at node-local scratch (1.7 TB of NVMe, private to the job) both
# removes the dependency on a shared quota and makes the unpack faster.
export SINGULARITY_TMPDIR="${SCRATCH}/sing_tmp"
export SINGULARITY_CACHEDIR="${SCRATCH}/sing_cache"
export SINGULARITY_LOCALCACHEDIR="${SCRATCH}/sing_local"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_LOCALCACHEDIR}"

touch "${COMPLETION_LOG}"
START=$(((SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1))
END=$((SLURM_ARRAY_TASK_ID * CHUNK))

sed -n "${START},${END}p" "${INPUT_FILE}" | while read -r accession; do
	[[ -n ${accession} ]] || continue
	if grep -qxF "${accession}" "${COMPLETION_LOG}"; then continue; fi

	gz=$(ls "${STRUCTURES}/AF-${accession}-F1-model_"*.cif.gz 2> /dev/null | head -1)
	[[ -n ${gz} ]] || continue
	cif="${SCRATCH}/${accession}.cif"
	gunzip -c "${gz}" > "${cif}" || continue

	pqr="${SCRATCH}/${accession}.pqr"
	"${CODE}/mmcifConvert.py" "${cif}" --output "${pqr}" --to-pqr || {
		rm -f "${cif}"
		continue
	}

	in="${SCRATCH}/${accession}.in"
	out="${SCRATCH}/${accession}.out"

	# Size the grid to the molecule instead of using a fixed box.
	#
	# The settings inherited from the other project fix dime at 97 and grid at 0.33 A, which
	# is a 32 A box. J-domain proteins routinely span far more than that - the first test
	# structure reached 67 A - and APBS responds by silently *ignoring every atom off the
	# mesh*. It still exits zero and still prints an energy, so a truncated molecule produces
	# a plausible-looking number with no warning in the result. Those settings presumably
	# suited that project's proteins; they do not suit these.
	#
	# cglen is the coarse box: molecular extent plus 20 A of solvent padding, which is the
	# usual margin for the boundary condition to be reasonable. dime is then chosen as the
	# smallest 2^k+1 value giving at least ~0.5 A spacing, since APBS multigrid requires
	# dimensions of that form.
	read -r EX EY EZ < <(
		awk '/^ATOM/ {
			x=$(NF-4); y=$(NF-3); z=$(NF-2)
			if (n++ == 0) { x0=x1=x; y0=y1=y; z0=z1=z }
			if (x<x0) x0=x; if (x>x1) x1=x
			if (y<y0) y0=y; if (y>y1) y1=y
			if (z<z0) z0=z; if (z>z1) z1=z
		} END { printf "%.1f %.1f %.1f\n", x1-x0+20, y1-y0+20, z1-z0+20 }' "${pqr}"
	)
	[[ -n ${EX} ]] || continue

	pick_dime() {
		local want=$1 d
		for d in 33 65 97 129 161 193 225 257 289 321; do
			if (($(echo "${d} * 0.5 >= ${want}" | bc -l))); then
				printf '%s' "${d}"
				return
			fi
		done
		printf '321'
	}
	DX=$(pick_dime "${EX}")
	DY=$(pick_dime "${EY}")
	DZ=$(pick_dime "${EZ}")

	# Same numerical settings as the source pipeline - lpbe, pdie 1.0, sdie 78.54, srad
	# 1.4, temp 298.15 - so the energies stay comparable. Only the grid differs, and it
	# has to: a fixed box silently truncates any protein larger than it.
	cat <<- APBSEOF > "${in}"
		read
		  mol pqr ${pqr}
		end
		elec name solv
		  mg-auto
		  dime ${DX} ${DY} ${DZ}
		  cglen ${EX} ${EY} ${EZ}
		  fglen ${EX} ${EY} ${EZ}
		  cgcent mol 1
		  fgcent mol 1
		  mol 1
		  lpbe
		  bcfl mdh
		  pdie 1.0
		  sdie 78.54
		  chgm spl2
		  srfm mol
		  srad 1.4
		  swin 0.3
		  sdens 10.0
		  temp 298.15
		  calcenergy total
		  calcforce no
		end
		elec name ref
		  mg-auto
		  dime ${DX} ${DY} ${DZ}
		  cglen ${EX} ${EY} ${EZ}
		  fglen ${EX} ${EY} ${EZ}
		  cgcent mol 1
		  fgcent mol 1
		  mol 1
		  lpbe
		  bcfl mdh
		  pdie 1.0
		  sdie 1.0
		  chgm spl2
		  srfm mol
		  srad 1.4
		  swin 0.3
		  sdens 10.0
		  temp 298.15
		  calcenergy total
		  calcforce no
		end
		print elecEnergy solv - ref end
		print elecEnergy ref end
		quit
	APBSEOF

	if singularity exec "${SIF}" apbs "${in}" > "${out}" 2> "${SCRATCH}/apbs.err"; then
		"${CODE}/apbs_scraper.py" "${out}" \
			--output "${OUTPUT_DIR}/jdp_energies_${SLURM_ARRAY_TASK_ID:-0}.tsv" \
			--protein-name "${accession}" &&
			printf "%s\n" "${accession}" >> "${COMPLETION_LOG}"
	else
		# One line per failure, so a run that produces no energies says why.
		printf "%s\t%s\n" "${accession}" \
			"$(tr '\n' ' ' < "${SCRATCH}/apbs.err" 2> /dev/null | cut -c1-200)" \
			>> "${OUTPUT_DIR}/failures_${SLURM_ARRAY_TASK_ID:-0}.tsv"
	fi
	rm -f "${cif}" "${pqr}" "${in}" "${out}" "${SCRATCH}/apbs.err"
done
