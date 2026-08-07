# Curated reference JDPs

`reference_domain_store.json` is a small domain store fetched with:

```bash
fetch-protein-domains P08622 P31689 P25685 P36659 P25686 Q9H3Z4 Q9NVH1 \
  -o data/reference_jdps/reference_domain_store.json
```

`reference_classes.json` maps each accession to its literature class. `analyze-domain-layout`
segments these proteins with the same region router used for query proteins and uses their
unalignable (IDR) regions as SHARK comparison targets.

| Accession | Protein | Class | Why it is here |
|-----------|---------|-------|----------------|
| P08622 | *E. coli* DnaJ | A | Canonical bacterial class A: J-domain, G/F-rich, zinc-finger-like, C-terminal domain |
| P31689 | Human DNAJA1 | A | Canonical eukaryotic class A |
| P25685 | Human DNAJB1 | B | Canonical class B: J-domain + G/F-rich + C-terminal domain, no zinc finger |
| P25686 | Human DNAJB2 | B | Class B with a membrane-anchored isoform |
| P36659 | *E. coli* CbpA | B | Bacterial class B-like (no zinc-finger-like region) |
| Q9H3Z4 | Human DNAJC5 | C | Class C, palmitoylated, J-domain plus cysteine-string region |
| Q9NVH1 | Human DNAJC11 | C | Class C with a large non-JDP C-terminal region |

Refresh the store with the command above when InterPro annotations change; the class map is
curated by hand and should only change with literature support.
