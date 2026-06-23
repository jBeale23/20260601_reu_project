# REU Data Fetching

Async Python tools for fetching protein data from the [InterPro](https://www.ebi.ac.uk/interpro/) REST API. The project includes two fetch scripts:

| Script | InterPro entry | Description |
|--------|----------------|-------------|
| `fetch-architectures-dnaj` | IPR001623 (DnaJ/HSP40) | Fetches proteins for the top *N* domain architectures |
| `fetch-proteins-dnak` | IPR012725 (DnaK) | Fetches all proteins for a single entry (no architecture groups) |

Both scripts write JSON output to disk. No deduplication is performed — proteins are stored exactly as returned by the API.

## Requirements

- Python **3.12+**
- Network access to `www.ebi.ac.uk`

## Installation

Clone the repository and install the package in **editable** mode so local code changes take effect immediately and console commands are registered.

### pip (recommended)

```bash
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest and pytest-asyncio for running the test suite.

### Make

On Linux or macOS:

```bash
make install
```

This runs `pip install -e ".[dev]"`.

### Conda

```bash
conda env create -f environment.yaml
conda activate reu_project
pip install -e .
```

The conda environment installs runtime and test dependencies via pip. Run `pip install -e .` afterward to register the package and console commands.

## Running the fetch scripts

After installation, use the console commands:

```bash
fetch-proteins-dnak --help
fetch-architectures-dnaj --help
```

Without installing, you can run the modules directly from the project root:

```bash
python -m data_fetching.fetch_proteins_dnak --help
python -m data_fetching.fetch_architectures_dnaj --help
```

### fetch-proteins-dnak (IPR012725 / DnaK)

Fetches all proteins matching InterPro entry IPR012725.

```bash
fetch-proteins-dnak
```

**Default output:** `ipr012725_proteins.json`

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--page-size` | Results per API page | `200` |
| `-o`, `--output` | Output JSON file path | `ipr012725_proteins.json` |
| `-k`, `--checkpoint` | Checkpoint file for resume | `ipr012725_checkpoint.json` |
| `-m`, `--max-percent` | Percent threshold for soft count notification | `0.01` (1%) |
| `-a`, `--max-abs` | Absolute threshold for hard count warning | `100` |
| `-t`, `--timeout` | Request timeout in seconds | `300` |

**Checkpoint / resume:** If a fetch is interrupted (for example by a timeout mid-pagination), progress is saved to the checkpoint file. Re-run the same command to resume from the last cursor. The checkpoint file is deleted automatically when the fetch completes successfully.

**Example:**

```bash
fetch-proteins-dnak -o my_proteins.json -t 600
```

### fetch-architectures-dnaj (IPR001623 / DnaJ)

Fetches proteins for the first *N* domain architecture groups of IPR001623. Proteins that appear in multiple architectures are stored multiple times; each record is flagged with `appears_in_architecture_count`.

```bash
fetch-architectures-dnaj
```

**Default output:** `ipr001623_domain_architectures_no_dedup.json`

| Flag | Description | Default |
|------|-------------|---------|
| `-n`, `--n-architectures` | Number of architectures to fetch | `20` |
| `-u`, `--arch-url` | Custom architecture list API URL | InterPro IPR001623 endpoint |
| `-c`, `--concurrency` | Max concurrent architecture fetches | `5` |
| `-o`, `--output` | Output JSON file path | `ipr001623_domain_architectures_no_dedup.json` |

**Example:**

```bash
fetch-architectures-dnaj -n 10 -o my_architectures.json
```

## Output format

### DnaK (`fetch-proteins-dnak`)

```json
{
  "entry_accession": "IPR012725",
  "proteins_reported": 100000,
  "total_proteins_fetched": 100000,
  "is_partial": false,
  "proteins": [],
  "note": "..."
}
```

### DnaJ (`fetch-architectures-dnaj`)

```json
{
  "architectures": [
    {
      "ida": "PF00226:IPR001623",
      "ida_id": "...",
      "unique_proteins_reported": 98256,
      "proteins_fetched": 99077,
      "is_partial": false,
      "proteins": [{"appears_in_architecture_count": 2}]
    }
  ],
  "total_proteins_fetched": 225610,
  "note": "..."
}
```

Fetched JSON files are gitignored by default.

## AlphaFold structure download (Rockfish + AlphaFoldFetch)

After fetching proteins from InterPro, you can download predicted structures from the [AlphaFold Database](https://alphafold.ebi.ac.uk/) using [AlphaFoldFetch](https://github.com/mansanlab/alphafoldfetch) (`affetch`).

### 1. Extract UniProt accessions

```bash
fetch-proteins-dnak -o ipr012725_proteins.json
python scripts/extract_uniprot_ids.py ipr012725_proteins.json -o accessions.txt
```

This also works on DnaJ architecture output (`ipr001623_domain_architectures_no_dedup.json`).

### 2. Set up AlphaFoldFetch on Rockfish (once)

```bash
conda create -n affetch python=3.12 -y
conda activate affetch
pip install AlphaFoldFetch
```

### 3. Prepare work directory on Rockfish

Copy accessions into the work directory (the SLURM script creates missing directories and log files automatically):

```bash
WK_DIR="${HOME}/scr4_sfried3/alphafoldfetch"
cp accessions.txt "${WK_DIR}/incomplete_accessions.txt"
```

If `incomplete_accessions.txt` is not present when the job starts, the script creates an empty file and array tasks exit without downloading anything.

### 4. Submit the SLURM array job

```bash
PROJECT_DIR="${HOME}/repositories/rockfish-projects/reu_project"
WK_DIR="${HOME}/scr4_sfried3/alphafoldfetch"
cd "${PROJECT_DIR}"
N=$(wc -l < "${WK_DIR}/incomplete_accessions.txt")
sbatch --array=1-"${N}"%128 scripts/slurm/affetch_rockfish.sh
```

Each array task downloads structures for **one** UniProt accession. Completed accessions are logged to `completed_accessions.txt`; re-submitting the same array skips finished IDs (same pattern as the Rockfish APBS example).

Optional environment variables for the job script:

| Variable | Default | Description |
|----------|---------|-------------|
| `FILE_TYPE` | `pcz` | `affetch -f` format: `p`=PDB, `c`=CIF, `z`=gzip |
| `MODEL_VERSION` | `6` | AlphaFold model version |
| `CONDA_ENV` | `affetch` | Conda environment name |

Structures are written to `${WK_DIR}/structures/` as `AF-<accession>-F1-model_v6.pdb.gz` (and `.cif.gz` by default).

## Development

### Run tests

```bash
pytest tests/ -v
```

Or via Make:

```bash
make pytest
```

### Pre-commit

Install hooks once:

```bash
pre-commit install
```

Run all checks manually:

```bash
pre-commit run --all-files
```

### Lint and format

```bash
ruff check .
ruff format .
```

## Project layout

```
data_fetching/
  fetch_architectures_dnaj.py   # DnaJ architecture fetcher
  fetch_proteins_dnak.py        # DnaK protein fetcher
  fetch_types.py                # Shared type aliases
  utils.py                      # Shared HTTP, logging, output, and checkpoint helpers
scripts/
  extract_uniprot_ids.py        # UniProt ID extraction for AlphaFoldFetch
  slurm/
    affetch_rockfish.sh         # Rockfish array job for affetch
tests/
  test_fetch_dnaj.py
  test_fetch_dnak.py
  test_extract_uniprot_ids.py
pyproject.toml                  # Package metadata, dependencies, and console script entry points
```

Shared logic (HTTP retries, logging, JSON output, count validation, checkpoint I/O) lives in `data_fetching/utils.py`. Shared types live in `data_fetching/fetch_types.py`.
