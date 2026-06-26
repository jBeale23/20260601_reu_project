# Pocket definition validation (1DKX)

This document records experimental validation of the DnaK SBD peptide-binding pocket definition in [`data/pocket_refs/dnak_sbd_pocket.yaml`](../data/pocket_refs/dnak_sbd_pocket.yaml).

## Source structure

| Item | Value |
|------|-------|
| PDB | [1DKX](https://www.rcsb.org/structure/1DKX) |
| Protein chain | A (SBD fragment, resnum 389–607) |
| Peptide chain | B (NRLLLTG heptapeptide) |
| Shell radius | 8.0 Å (Cα–Cα) |
| Literature | Zhu X, et al. *Science* 1996 |

## Round-trip contact validation

Contact residues were **independently re-derived** from the experimental structure by selecting protein Cα atoms within 8 Å of any peptide Cα. The derived set matches the 17 residues in the YAML definition exactly:

```
402, 403, 404, 425, 426, 427, 428, 429, 430, 433, 434, 435, 436, 437, 438, 458, 468
```

Automated test: `tests/test_charge.py::test_1dkx_derived_contacts_match_yaml` (fixture: `tests/fixtures/1DKX.pdb`).

## AlphaFold vs experimental (reference accession)

For the reference accession `P0A6Y8`, AlphaFold model v6 reproduces the pocket mapping with:

- 17/17 contact residues mapped
- `confidence_tier: high`
- `mapping_mode: sbd_local`
- Zero charge delta vs itself

Full-structure Poisson–Boltzmann comparison (APBS) is deferred until mapping quality is validated at scale.

## v4 dev-set summary (mapping vs conservation)

v4 separates **mapping confidence** (structural/SBD placement) from **conservation score** (sequence similarity at contacts). When AlphaFold models cannot be superposed reliably (SBD RMSD ≥10 Å), mapping tier falls back to SBD-domain alignment quality.

| Accession | Mapping | Conservation | Contact Δ | DRMSD | Flags |
|-----------|---------|--------------|-----------|-------|-------|
| P0A6Y8 | high | high | 0 | 0.0 | — |
| P08113 | high | low | +2 | 8.7 | charge_inversion_candidate |
| P11142 | high | medium | 0 | 1.4 | — |
| P42943 | medium | low | +1 | — | partial mapping (76%) |
| P61889 | low | low | +5 | 8.3 | full_length_alignment |

Yeast (`P08113`) is flagged `charge_inversion_candidate`: mapping is reliable (SBD-domain align) but contact sequences are divergent — the expected pattern for testing charge inversion.

### Structural superposition notes

Cross-structure Kabsch superposition of AlphaFold apo models often yields high SBD RMSD (~25 Å) due to domain orientation differences. v4 therefore also reports:

- `contact_drmsd` — rotation-invariant pocket geometry similarity (pairwise Cα distance matrix RMSE)
- `mean_contact_ca_distance` — after Kabsch, when superposition is reliable (SBD RMSD &lt; 10 Å)
- `unreliable_superposition` quality flag when SBD RMSD ≥ 10 Å

## v3 dev-set summary (after SBD-domain alignment)

| Accession | Tier | Mode | Contact Δ | Shell Δ | Contact identity | Notes |
|-----------|------|------|-----------|---------|------------------|-------|
| P0A6Y8 | high | sbd_local | 0 | 0 | 100% | Reference |
| P08113 | low | sbd_local | +2 | +5 | 18% | Strongest biological signal; low tier due to divergent contact sequences |
| P11142 | low | sbd_local | 0 | 0 | 47% | Charge neutral vs reference |
| P42943 | low | sbd_local | +1 | −2 | 38% | Partial contact mapping (76%) |
| P61889 | low | full_length | +5 | +5 | 24% | Falls back to full-length; many contact mismatches |

### Sensitivity (parameter sweep)

Sweep over shell radius (6/8/10 Å), min pLDDT (60/70/80), and SBD window ±10 residues:

| Accession | Stable contact Δ? | Stable shell Δ? |
|-----------|-------------------|-----------------|
| P08113 | Yes (+2) | No (4 to +6) |
| P0A6Y8 | Yes | Yes |
| P11142 | Yes | No |
| P42943 | Yes (+1) | No |
| P61889 | Yes (+5) | No |

Yeast (`P08113`) contact charge shift (+2) is **stable** across parameter sweeps; shell charge is more parameter-sensitive, as expected.

## PyMOL validation exports

```bash
analyze-pocket-charge data/dev_structures/AF-P08113-F1-model_v6.pdb \
  --export-pocket-residues pocket_residues_P08113.csv

analyze-pocket-charge data/dev_structures/AF-P08113-F1-model_v6.pdb \
  --export-contact-attribution contact_attribution_P08113.csv
```

Use the CSVs to color contact vs shell residues and inspect per-contact charge contributions in PyMOL.
