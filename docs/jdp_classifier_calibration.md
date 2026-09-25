# JDP classifier calibration

Spot-check expectations for the two classification paths in this repo:

- **`classify-jdp`** — rule-based class A/B/C from the domain architecture (plus HPD, TM/signal, layout tags).
- **`analyze-domain-layout`** — full domain complement + disorder architecture + SHARK similarity, producing a class, a subclass, and a novel-category score.

## Class rules

The zinc-finger-like cysteine-rich region is the class A/B discriminator (Kampinga & Craig; see the FEBS review linked from the README):

| Rule | Class |
|------|-------|
| J-domain + zinc-finger-like region (`PF00684` / `PF00569` / `IPR001305`) | **A** |
| J-domain + C-terminal substrate-binding domain (`PF01556` / `IPR002939`) and/or G/F-rich region, no zinc finger | **B** |
| J-domain present, neither of the above | **C** |
| No J-domain | **unknown** |

An earlier revision required "≥3 domains + (C-terminal domain **or** zinc finger)" for class A, which mislabelled canonical class B proteins such as human DNAJB1 (J-domain + C-terminal domain, no zinc finger). Both classifiers now use the rules above.

## Automated checks

`tests/test_jdp_classifier.py` (architecture rules):

| Case | Expected |
|------|----------|
| `PF00226-PF01556-PF00684` (J + CTD + zinc finger) | Class **A** |
| `PF00226-PF01556` (J + CTD, no zinc finger) | Class **B** |
| `PF00226-PF09320` (J + G/F-rich) | Class **B** |
| `PF00226-PF00564` (2 domains, no G/F-rich, no CTD) | Class **C** (`atypical_multi_domain`) |
| `PF00226` only | Class **C** (`j_domain_only`) |
| Domain store supplies a richer architecture than the fetch IDA | `architecture_source=domain_store` |

`tests/test_reference_jdps.py` (layout classifier, run against the bundled reference store):

| Accession | Protein | Expected class | Expected subclass |
|-----------|---------|----------------|-------------------|
| P08622 | *E. coli* DnaJ | **A** | `a_canonical` |
| P31689 | Human DNAJA1 | **A** | `a_canonical` |
| P25685 | Human DNAJB1 | **B** | `b_canonical` |
| P25686 | Human DNAJB2 | **B** | `b_gf_rich_no_ctd` |
| P36659 | *E. coli* CbpA | **B** | `b_canonical` |
| Q9H3Z4 | Human DNAJC5 | **C** | `c_j_domain_only` |
| Q9NVH1 | Human DNAJC11 | **C** | `c_atypical_multi_domain` |

None of these may be flagged as a novel-category candidate; the test suite fails if any is.

```bash
pytest tests/test_jdp_classifier.py tests/test_reference_jdps.py -v
```

### Backend independence

The disorder predictor decides where region boundaries fall; it must not decide the class.
When metapredict is installed, `tests/test_reference_jdps.py` runs the whole reference set
through **both** backends and asserts identical class and subclass calls, and
`tests/test_domain_disorder.py` checks the real metapredict boundary conversion against the
1-based inclusive coordinates the region router expects.

This test exists because it caught a real defect: G/F-rich detection originally scored whole
regions, so when metapredict merged DNAJB2's (`P25686`) G/F block into a longer IDR, the
composition was diluted below threshold and the protein flipped from **B** to **C**.
Detection now scans 30-residue windows, which is boundary-independent.

Verified agreement on all seven references across metapredict V3 vs FoldIndex, SHARK vs the
BLOSUM k-mer fallback, and the production pairing (metapredict + SHARK) together.

A second real-backend defect surfaced the same way: both scorers look residues up in a
substitution matrix keyed by upper-case letters, so a lower-case region scored 0.0 as if
unrelated — inflating its novelty score. Sequences are now upper-cased before scoring, and
`tests/test_domain_shark.py` asserts case-insensitivity on both backends.

Similarity scores are **not** comparable across backends (identical regions score ~0.86
under SHARK and 1.00 under the fallback), which is why `shark_backend` is recorded in every
row. Both scales separate related from unrelated regions well clear of the 0.35 threshold.

With SHARK installed, each reference protein's closest unalignable region belongs to a
reference **of its own class** — the property the novelty score depends on.

**Note on accessions:** `P0ACJ8` is *E. coli* CRP (catabolite activator protein), **not** DnaJ. Earlier revisions of this document and of the test suite used it as the "classic DnaJ" reference. The correct *E. coli* DnaJ accession is **`P08622`**.

## Manual validation on a real fetch

```bash
fetch-architectures-dnaj -o ipr001623_domain_architectures_no_dedup.json
fetch-protein-domains --from-fetch-json ipr001623_domain_architectures_no_dedup.json -o protein_domains.json

classify-jdp ipr001623_domain_architectures_no_dedup.json \
  --domain-json protein_domains.json \
  -o data/jdp_classifications/jdp_classifications.csv

analyze-domain-layout protein_domains.json -o domain_layout_results
```

| Accession | Organism | Expected class | Notes |
|-----------|----------|----------------|-------|
| P08622 | *E. coli* DnaJ | **A** | Classic bacterial DnaJ; HPD expected; no TM/signal |
| P31689 | Human DNAJA1 | **A** | Type A human JDP |
| P25685 | Human DNAJB1 | **B** | Type B; C-terminal domain, no zinc finger |
| P36659 | *E. coli* CbpA | **B** | Bacterial class B-like |
| Q9H3Z4 | Human DNAJC5 | **C** | J-domain only; check `layout_predicted_subclass` |

Where the two classifiers disagree, the difference is usually annotation coverage: `classify-jdp` sees Pfam matches only, while `analyze-domain-layout` also uses CDD/SMART/Gene3D signatures and compositional evidence (for example DNAJB2, whose C-terminal region has no Pfam match but whose G/F-rich block is detected compositionally). Both columns are kept side by side in `merge-all-features` output (`jdp_predicted_class` and `domain_layout_predicted_class`).

## Novel-category screening

`layout_novelty_score` sums four independent mismatches with the reference templates (no HPD 0.25, J-domain not N-terminal 0.15, no canonical partner domain 0.20, unannotated partner domain 0.15, SHARK-dissimilar 0.25) and flags `novel_class_candidate` at ≥ 0.5. Sanity expectations:

- Reference JDPs: score 0.0–0.35, never flagged.
- A J-domain protein with no HPD, an internal J-domain, and no recognizable partner module: flagged.
- Proteins with no J-domain: class `unknown`, score 0.0, never flagged (the score only describes JDPs).

Check candidates before trusting them: `layout_evidence_tags` records which terms fired, and `shark_backend` records whether the similarity term came from real SHARK or the k-mer fallback.

## Localization sources

1. **InterPro entries** on the protein record (including member-database signatures via their integrated InterPro parent).
2. **UniProt features JSON** when `--no-fetch` is not set and InterPro lacks TM/signal annotations.

Curated InterPro/Pfam IDs live in `jdp_classifier/constants.py` (`TRANSMEMBRANE_*`, `SIGNAL_PEPTIDE_*`).

## Merge integration

```bash
merge-all-features \
  --dnaj-json ipr001623_domain_architectures_no_dedup.json \
  --jdp-csv data/jdp_classifications/jdp_classifications.csv \
  --domain-csv domain_layout_results/domain_layout_features.csv \
  -o data/merged_features/dnaj_jdp.csv
```

Classifier columns arrive as `jdp_*`, layout columns as `domain_*`, and the row-level `novel_class_candidate` flag plus `classification_tags` (`layout_<subclass>`, `novel_class_candidate`) make candidates filterable in one pass.
