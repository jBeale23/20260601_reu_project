# JHU BioREU JDP Classification Project

The project goal is building a **protein ID system** for J-domain proteins. Instead of saying “this unknown protein kind of looks like something from *E. coli* or humans,” we want a systematic way to look at its parts and predict what job it probably does in the Hsp70 chaperone system ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

## Core background

Proteins are chains of amino acids that fold into 3D shapes. Those shapes let proteins do cellular jobs: catalyzing reactions, carrying signals, building structures, or binding other molecules.

Sometimes proteins fold incorrectly, partly unfold, or clump together. **Chaperones** are helper proteins that prevent this damage or help proteins recover ([PubMed](https://pubmed.ncbi.nlm.nih.gov/35729039/)).

**Hsp70** is one of the most important chaperones. It works like a reusable clamp: it grabs exposed sticky parts of proteins, uses ATP energy, and releases them so they get another chance to fold correctly ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

**J-domain proteins (JDPs)** — Hsp40 is a familiar example — tell Hsp70 where to act. Hsp70 is the powerhouse; JDPs provide much of the targeting logic ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.15.618527v1.full.pdf)). JDPs recognize misfolded, unfolded, or aggregating proteins and recruit **ATP-bound Hsp70**. In the ATP-bound state, Hsp70’s **nucleotide-binding domain (NBD)** and **substrate-binding domain (SBD)** are tightly coupled; contact with the J-domain stimulates ATP hydrolysis so Hsp70 can grip and release client proteins in a controlled cycle ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

## Acronym guide

| Acronym | Meaning | Simple meaning |
|---------|---------|----------------|
| JDP | J-domain protein | A protein that helps Hsp70 work. |
| JD | J-domain | The small part of a JDP that contacts Hsp70. |
| Hsp70 | Heat shock protein 70 | A major chaperone that helps proteins fold or recover. |
| Hsp40 | Heat shock protein 40 | Older/common name for many JDPs. |
| DnaK | Bacterial Hsp70 | The *E. coli* version of Hsp70. |
| DnaJ | Bacterial JDP/Hsp40 | The classic JDP used as the reference model. |
| ATP | Adenosine triphosphate | The cell’s energy currency. |
| ADP | Adenosine diphosphate | The lower-energy product after ATP is used. |
| NEF | Nucleotide exchange factor | A protein that helps Hsp70 reset for another cycle. |
| HPD | Histidine-proline-aspartate | A key three-amino-acid motif in real J-domains. |
| G/F-rich | Glycine/phenylalanine-rich | A flexible region common in class A and B JDPs. |
| DNAJA | Class A human-style JDP group | DnaJ-like JDPs with many classic domains. |
| DNAJB | Class B human-style JDP group | J-domain plus G/F-rich region but less complete DnaJ-like architecture. |
| DNAJC | Class C human-style JDP group | Very diverse JDPs that do not fit class A or B. |
| ER | Endoplasmic reticulum | A cell compartment where secreted and membrane proteins fold. |
| TM | Transmembrane | A protein segment that crosses a membrane. |
| TPR | Tetratricopeptide repeat | A repeated protein-binding module found in some chaperone-related proteins. |
| DUF | Domain of unknown function | A predicted domain whose job is not yet well understood. |
| HMM | Hidden Markov model | A computational pattern detector used to find protein domains. |
| Pfam / InterPro | Protein domain databases | Tools/databases used to label domains in protein sequences. |

Key references for this section: [FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359), [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.10.15.618527v1.full.pdf), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35729039/).

## What a domain is

A **domain** is a reusable part of a protein — like a LEGO block with a specific shape and job.

A protein’s **domain architecture** is which domains it has and in what order. For JDPs this matters because the J-domain activates Hsp70, while other domains often decide which substrates the JDP recognizes, where it localizes, and which pathway it joins ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.15.618527v1.full.pdf)).

This project is not only asking “Does this protein have a J-domain?” It asks **“What kind of JDP is this, based on the full layout of its components?”**

## JDP classes

The simplest classification is **class A**, **class B**, and **class C**, based on similarity to classic bacterial DnaJ ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

| Class | Simple signature | Simple interpretation |
|-------|------------------|----------------------|
| Class A / DNAJA | N-terminal J-domain, G/F-rich region, client-binding domains, zinc-finger-like region, and dimerization region | Most similar to classic DnaJ; often a general protein-folding helper. |
| Class B / DNAJB | N-terminal J-domain and G/F-rich region but lacks some class A features (especially the zinc-finger-like region) | Similar to class A, but often more specialized. |
| Class C / DNAJC | J-domain present but architecture does not match class A or B | A large mixed category; often specialized for particular pathways. |

Treat “class C” cautiously: it groups many specialized JDPs that are not yet fully subdivided ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

## Classifier goals

The long-term goal is an **unbiased classifier** — one that does not rely only on classic model organisms (*E. coli*, yeast, humans) but can handle unfamiliar proteins from non-model species ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.15.618527v1.full.pdf)).

For each protein, the classifier should eventually answer:

- Does it have a real J-domain?
- Does the J-domain contain the important HPD motif?
- Where is the J-domain: beginning, middle, or end?
- Does it have a G/F-rich region?
- Does it have client-binding domains?
- Does it have membrane-spanning regions?
- Does it have signal peptides that target a compartment?
- Does it resemble class A, class B, class C, or a more specific subclass?
- Is the classification high-confidence or uncertain?

A useful output goes beyond “class C” — for example: “High-confidence membrane-associated class C JDP with an HPD-containing J-domain and a transmembrane segment, likely recruiting Hsp70 to a membrane-localized process” ([FEBS review](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.70359)).

JDPs customize where and how Hsp70 acts: newly made proteins, damaged proteins, aggregates, membranes, organelles, ribosomes, or degradation pathways ([PubMed](https://pubmed.ncbi.nlm.nih.gov/35729039/)).

## Data fetching

This repository’s current code collects reference and homolog data from [InterPro](https://www.ebi.ac.uk/interpro/) for the bacterial JDP and Hsp70 entries used as anchors in the classification work:

| Script | InterPro entry | Description |
|--------|----------------|-------------|
| `fetch-architectures-dnaj` | IPR001623 (DnaJ/HSP40) | Fetches proteins for the top *N* domain architectures |
| `fetch-proteins-dnak` | IPR012725 (DnaK) | Fetches all proteins for a single entry (no architecture groups) |

Both scripts write JSON output to disk. No deduplication is performed, so proteins are stored exactly as returned by the API.
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
extract-uniprot-ids --help
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
extract-uniprot-ids ipr012725_proteins.json -o accessions.txt
```

This also works on DnaJ architecture output (`ipr001623_domain_architectures_no_dedup.json`).

### 2. Set up AlphaFoldFetch on Rockfish (once)

```bash
conda env create -f scripts/slurm/conda_env.yaml
conda activate affetch
```

### 3. Prepare work directory on Rockfish

Copy accessions into the work directory (the SLURM script creates missing directories and the completion log automatically):

```bash
WK_DIR="${HOME}/scr4_sfried3/alphafoldfetch"
cp accessions.txt "${WK_DIR}/incomplete_accessions.txt"
```

The array job exits immediately with an error if `incomplete_accessions.txt` is missing.

### 4. Submit the SLURM array job

```bash
bash scripts/slurm/submit_affetch_rockfish.sh
```

The launcher counts pending accessions (input minus completion log) and passes `--array=1-N%128` to `sbatch` automatically.

Optional environment variables for the job script:

| Variable | Default | Description |
|----------|---------|-------------|
| `FILE_TYPE` | `pcz` | `affetch -f` format: `p`=PDB, `c`=CIF, `z`=gzip |
| `MODEL_VERSION` | `6` | AlphaFold model version |
| `CONDA_ENV` | `affetch` | Conda environment name (from `scripts/slurm/conda_env.yaml`) |

Each array task downloads structures for **one** UniProt accession. Completed accessions are logged to `completed_accessions.txt`; failed downloads are logged to `failed_accessions.txt` so they can be retried or inspected separately. Re-submitting via the launcher skips finished IDs (same pattern as the Rockfish APBS example).

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
    submit_affetch_rockfish.sh  # Launcher: sets array bounds and submits job
    conda_env.yaml              # Pinned conda env for affetch on Rockfish
tests/
  test_fetch_dnaj.py
  test_fetch_dnak.py
  test_extract_uniprot_ids.py
pyproject.toml                  # Package metadata, dependencies, and console script entry points
```

Shared logic (HTTP retries, logging, JSON output, count validation, checkpoint I/O) lives in `data_fetching/utils.py`. Shared types live in `data_fetching/fetch_types.py`.
