# JDP classifier v1 calibration

Spot-check expectations for the rule-based classifier after v1 changes (G/F-rich class B, TM/signal localization, layout tags).

## Automated checks

Unit tests in `tests/test_jdp_classifier.py` cover:

| Case | Expected |
|------|----------|
| `PF00226-PF01556-PF00684` (3 domains, DnaJ C + Zn) | Class **A** |
| `PF00226-PF09320` (J + G/F-rich) | Class **B** |
| `PF00226-PF00564` (2 domains, no G/F-rich) | Class **C** (`atypical_multi_domain`) |
| `PF00226` only | Class **C** (`j_domain_only`) |
| P0ACJ8-like synthetic record | Class **A**, HPD, no TM/signal |

Run:

```bash
py -3.12 -m pytest tests/test_jdp_classifier.py -v
```

## Reference accessions (manual validation on fetch JSON)

When a DnaJ architecture fetch JSON is available locally:

```bash
classify-jdp ipr001623_domain_architectures_no_dedup.json \
  -o data/jdp_classifications/jdp_classifications.csv
```

| Accession | Organism | Expected class | Notes |
|-----------|----------|----------------|-------|
| P0ACJ8 | *E. coli* DnaJ | **A** | Classic bacterial DnaJ; HPD expected; no TM/signal |
| P25685 | Human DNAJA1 | **A** | Type A human JDP |
| P25686 | Human DNAJA2 | **A** | Type A human JDP |
| P25687 | Human DNAJB1 | **B** | Type B; requires G/F-rich Pfam in IDA |
| Q9UGP8 | Human DNAJC5 | **C** | Class C; check `layout_tags` for specificity |

Filter the output CSV by accession to confirm `predicted_class`, `has_gf_rich`, `has_transmembrane`, `has_signal_peptide`, and `layout_tags`.

## Localization sources

1. **InterPro entries** on each protein record (primary)
2. **UniProt features JSON** when `--no-fetch` is not set and InterPro lacks TM/signal annotations

Curated InterPro/Pfam IDs live in `jdp_classifier/constants.py` (`TRANSMEMBRANE_*`, `SIGNAL_PEPTIDE_*`).

## Merge integration

New columns flow into `merge-all-features` as `jdp_has_gf_rich`, `jdp_has_transmembrane`, `jdp_has_signal_peptide`, `jdp_localization_source`, and `jdp_layout_tags`.

```bash
merge-all-features \
  --dnaj-json ipr001623_domain_architectures_no_dedup.json \
  --jdp-csv data/jdp_classifications/jdp_classifications.csv \
  -o data/merged_features/dnaj_jdp.csv
```
