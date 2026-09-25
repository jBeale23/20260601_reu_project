#!/bin/bash -ue
#SBATCH --job-name=merge-store
#SBATCH --partition=shared
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=sfried3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Merge the rare-architecture proteins into the domain store.
#
# The store holds 181,526 proteins carrying exactly 20 distinct architecture ids - the
# top-20 architectures and nothing else. The remaining 4,135 architectures were fetched
# (36,175 proteins, 91 chunks in domain_results_new) but never merged, so every analysis so
# far has run on the top-20 set. Merging brings the store to 217,701.
#
# The existing store is kept rather than overwritten: it is what every completed result was
# computed against, and a comparison between the two is the only way to say what the rare
# architectures changed.
WK="${HOME}/scr4_sfried3/domain_layout_run"
VENV="${HOME}/layout_venv"

MERGED="${WK}/protein_domains_all_architectures.json"

printf "[%s] merging %s chunk(s) with the existing store\n" "$(date +%T)" \
	"$(find "${WK}/domain_results_new" -name '*.json' | wc -l)"

# Stage both inputs where --merge-stores can see them: it merges every .json in one
# directory, and the existing store lives elsewhere.
STAGE="${WK}/merge_inputs"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
ln -sf "${WK}/protein_domains_backfilled.json" "${STAGE}/000_existing_store.json"
for chunk in "${WK}"/domain_results_new/*.json; do
	ln -sf "${chunk}" "${STAGE}/$(basename "${chunk}")"
done

"${VENV}/bin/fetch-protein-domains" --merge-stores "${STAGE}" -o "${MERGED}"

"${VENV}/bin/python" - "${MERGED}" <<'PY'
import sys
from collections import Counter

from domain_layout.records import load_domain_store

store = load_domain_store(sys.argv[1])
idas = Counter(p.ida_accession for p in store.proteins.values())
print(f"merged store: {len(store.proteins):,} proteins")
print(f"distinct architectures: {len(idas):,}")
PY

rm -rf "${STAGE}"
printf "[%s] done\n" "$(date +%T)"
