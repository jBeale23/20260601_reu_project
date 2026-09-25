# Pipeline design, and the reason for every choice

Each section states what the stage does, what was chosen, and what measurement or argument
forced the choice. Where a default was inherited rather than decided, that is said.

---

## 1. What proteins enter the set

**Choice.** Every UniProt protein carrying InterPro IPR001623 (the J-domain), fetched by
domain architecture.

**Why architecture-first.** The alternative is one flat protein list. Fetching by
architecture keeps the grouping InterPro already computed, which is the unit the class
scheme is defined on and the unit the recurrence test needs.

**The boundary, and its history.** The dataset was the 20 most common architectures:
181,526 proteins, 73.1% of the 248,251 carrying the signature. That number was chosen
deliberately, but `page_size=20` was also fixed in the request URL and only the first page
was ever read — so `--n-architectures` could not have raised it later even if wanted. Page
size and dataset size are now separate parameters, with a test asserting they can never
again be the same number.

**Why the remaining 4,135 matter.** They hold 66,725 proteins and average ~16 members
each. Rare architecture is the definition of what a novel-category search looks for, so a
search restricted to common architectures was filtered against its own target.

---

## 2. Residue routing

**Choice.** Every residue is assigned to exactly one region, and each region to one of
`msa`, `shark`, or `skip`. Boundaries come from InterPro matches; the gaps between them are
split by metapredict disorder into IDR and non-IDR pieces.

**Why exactly one.** Overlapping assignment would double-count residues in every
downstream sum; unassigned residues would make the two layers describe different proteins.

**Measured outcome.** 32.76M residues to MSA, 31.83M to alignment-free — **51% / 49%**.
Roughly half of JDP sequence is unalignable, which is the empirical justification for
having two layers rather than one.

**Discontinuous domains keep their fragments.** A domain interrupted by an insertion is
stored as separate pieces rather than one span. This is correct for coordinates and caused
a downstream problem addressed in §4.

---

## 3. Disorder backend

**Choice.** metapredict where possible, FoldIndex otherwise, chosen **per protein** and
recorded per protein.

**Why per protein rather than per run.** metapredict cannot encode residues outside the 20
standard amino acids, and real UniProt sequences contain `X`. Batching without a screen let
a single unusual sequence downgrade its whole batch — on one run, 10 bad sequences pushed
1,250 of 2,000 proteins onto the fallback while the summary still claimed metapredict. With
per-sequence screening the full run used metapredict for 180,552 and FoldIndex for 974.

**Why report the backend at all.** A FoldIndex score is not a metapredict score. Printing
one under the other's name is the failure mode this project has hit repeatedly, so every
output row carries the backend that produced it.

---

## 4. The MSA layer

**Choice.** MAFFT, `--auto --anysymbol --quiet --op 3.0 --thread N`, on length-banded
groups, with a progressive fallback when the binary is absent.

**Each flag.**
- `--auto` selects the algorithm from input size; one command is then correct for a
  six-member family and a two-thousand-member one.
- `--anysymbol` is required, not cosmetic — these sequences contain X, U, B and Z, and
  without it MAFFT rejects or rewrites them.
- `--op 3.0` raises the gap-open penalty above MAFFT's default of 1.53, which is tuned for
  full-length proteins that may carry real long insertions. These are length-bounded single
  domains. Measured on 300 sampled J-domains: default gave 142 columns at 57.7% gaps, 3.0
  gave 113 at 46.9%.
- `--reorder` is deliberately **absent**; rows map back to accessions by position.
- Output is upper-cased, because MAFFT lower-cases in several modes and the charge tables
  are case-sensitive.

**Length banding, and the measurement that forced it.** `dnaj_c` is bimodal: 46,245 regions
near 27 residues beside 41,802 near 125, because a domain split by an inserted zinc finger
is stored as fragments alongside intact copies (§2). Aligning them together gave 92% gaps —
a matrix on which any conservation score describes the outliers' insertions rather than the
family. Groups are now capped at a 2.0 length ratio.

**Why 2.0, and why a ratio.** Arithmetic, not taste: aligning length *L* against length *kL*
leaves at least (1 − 1/k) of the matrix as gaps before any biology. Bounding the ratio
bounds the floor on the gap fraction.

| family | gaps before | gaps after (best–worst band) |
|---|---|---|
| dnaj_c | 0.921 | **0.466** – 0.774 |
| tpr | 0.960 | **0.172** – 0.902 |
| j_domain | 0.810 | **0.422** – 0.784 |
| other_domain | 0.956 | 0.713 – 0.830 |

`other_domain` stays high because it is a catch-all bucket, not a homologous family.
Banding cannot fix something that was never one family.

**Nothing is discarded.** Every region lands in exactly one band. Bands too small to align
are *reported* as such — a family that shatters into singletons is a finding about that
family, not something to hide.

**An earlier attempt that failed, kept as a warning.** Before banding, outliers were
trimmed by a median-based window. It declined on exactly the families that needed it
(dnaj_c, other_domain, tpr all exceeded a 25% drop limit), leaving them above 92% gaps. A
fix that does not fire where the problem is is not a fix.

---

## 5. The alignment-free layer

**Choice.** bio-shark (real SHARK), with a BLOSUM62 k-mer fallback, chosen **per pair**.

**Why alignment-free at all.** IDRs, G/F-rich blocks and linkers cannot be aligned;
attempting it produces confident nonsense. SHARK relates k-mers instead.

**Why per pair.** bio-shark's substitution matrix covers only the 20 standard residues.
Handed anything else it writes one error line per offending k-mer *pair* and drops those
pairs from the score. On the full run that produced **16,215,464 error lines, a 1.3 GB
log** — and a quietly degraded score. Screening those pairs to the k-mer backend cut the
log to 512 bytes and the run from 14h03m to 34 minutes.

**A second bug, quieter.** The vectorized scorer normalised by `sqrt` of a product of
self-scores. BLOSUM62 scores X against itself at **−1**, so an X-rich k-mer sums negative:
one negative gives NaN, and *two* give a positive product, so `sqrt` succeeds and returns a
confident-looking number derived from nothing. The scalar path had always refused both; the
vectorized path had not.

---

## 6. Grammar and complexity

**Choice.** Reduced-alphabet n-mer entropy, entropy rates, and compression complexity
measured against a composition-matched shuffle.

**Why a reduced alphabet.** 20 amino acids give 20^k k-mers, far too sparse to estimate a
distribution over above k=2. Chemical reduction collapses substitutions that preserve
biophysics, so a diverged homolog produces the same grammar.

**Why the shuffle control is not optional.** Raw compression ratio is dominated by length:
uniformly random sequence compresses to 1.32 of its size at 25 residues and 0.515 at 1500.
Thresholding that compares lengths, not grammars. Shuffling destroys order while preserving
length and exact composition, so `1 − C(x)/mean(C(shuffle(x)))` isolates arrangement.
Measured: random ≤0.02, compositionally-skewed-but-aperiodic ≤0.10, tandem repeat 0.40–0.93.

**Why two complexity tests, not one.** A homopolymer scores exactly 0 on the shuffle
control — every permutation of `QQQQ…` is itself. That case has near-zero first-order
entropy. Merging them into one score would hide both blind spots.

**LZMA preset 6, not 9|EXTREME.** Identical order scores to three decimals, 193 ms → 5.6 ms
per region. A 34× difference decides whether profiling a proteome takes minutes or hours.

**Bias correction.** Plug-in entropy from a short region is biased downward, so a
30-residue linker would look more ordered than a 300-residue one purely through sample
size. Every entropy carries the Miller-Madow correction.

**Conditional profile.** Cost of each window given everything before it, per residue.
Building it exposed two errors: a window at position zero has an empty prefix and was
charged the compressor's entire stream header (a constant 12.0 bits/residue that became the
peak of *every* sequence), and a peak-over-mean ratio explodes as the mean approaches zero —
which is exactly what repetitive G/F-rich regions do.

---

## 7. Charge, at two scales

**Choice.** Net charge, charged fraction, and windowed charge with a segregation statistic.
Histidine is left neutral (pKa ≈ 6).

**Why both scales.** Net charge over a whole protein averages a basic patch against an
acidic one. Three sequences with **identical net charge of zero**:

| | net | segregation |
|---|---|---|
| K-block then D-block | 0.0 | **0.82** |
| KDKDKD, same composition | 0.0 | **0.00** |
| R-block, spacer, E-block | 0.0 | **0.89** |

**Why divide by the charged fraction.** It separates "few charges spread thinly" from
"many charges gathered into opposing blocks", which a raw window spread confuses.

---

## 8. Structure

**Choice.** AlphaFold models, 142,948 of 181,526 proteins (78.8% coverage).

**Why this is the load-bearing evidence.** Every other signal is sequence-derived. When a
domain signature is absent, sequence cannot distinguish "the protein lacks the domain" from
"nothing annotated it". A structure can: an unannotated but present domain still models as
folded.

**Features and their justification.**
- **pLDDT bands** — below ~50 AlphaFold cannot place a residue that has no single place to
  be, making it an *independent* disorder signal from a different model on different
  training data than metapredict.
- **Burial** by neighbour count — surface charge drives binding, buried charge is
  structural; net charge answers neither.
- **Compactness**, Rg against a length-matched expectation, so it does not re-measure length.
- **Relative contact order** — mean sequence separation of contacts, the topology signal
  invisible to both composition and alignment.

**Deliberate omissions.** Accessibility is a neighbour count, not a true Shrake-Rupley
surface — isolated behind one function so a real calculation can replace it. Pocket
geometry is absent because honest cavity detection needs an algorithm, and a wrong pocket
volume is worse than none.

**Coverage is reported as a result.** 38,578 proteins have no model. A systematic gap —
models for model organisms, absent elsewhere — would bias every structural conclusion, so
the number is stated rather than assumed away.

---

## 9. Quality control

**Choice.** Exclude on UniProt's fragment flag, length below 100, or a single **domain**-type
signature covering ≥95% of the sequence.

**Why the fragment flag matters most.** 13,092 flagged fragments, of which **12,220 carry
no other warning sign** — they look complete to every length and coverage heuristic. A
truncated protein has "a J-domain and no partners" *by construction*, which is the exact
novel-candidate signature.

**Why domain-type only.** Family and homologous-superfamily signatures (NCBIfam, HAMAP,
PANTHER, CATH-Gene3D) span whole proteins by design. Counting them excluded textbook
full-length class A JDPs as truncated — on a 60-protein pilot, 20 exclusions became 2.

---

## 10. Validation

**Labels.** Reviewed UniProt subfamily names — the only labels not produced by this
pipeline. 129 proteins. **Effective n is ~39**: 122 of 129 sit in cross-genus ortholog
groups and 85% are mammals, so confidence intervals computed on 129 are optimistic.

**Nulls.** The obvious null — shuffle whole architectures — is degenerate: it preserves the
multiset that determines the candidate count, so it can never fail. It is kept with a test
asserting it detects nothing. The real null is a configuration model over the
protein-to-partner-domain graph, preserving both degree sequences.

**Homology baseline.** Profile HMMs are the standard method and what InterPro's own
signatures are built from, so a rule system that cannot beat `hmmsearch` has not earned its
complexity. Cross-validated out-of-fold, because a profile built from the protein it later
scores is not a weak baseline but an invalid one.

**Blocking.** Genus blocking removed 2,174 same-genus relatives → 0, and changed accuracy
by 0.000 — because the leakage is at *ortholog* level: human and mouse DNAJB1 are different
genera and ~95% identical. Taxonomy is the wrong blocking unit for this label set.

**Capacity.** A benchmark of *n* targets with annotation error ε can certifiably order at
most `n // (floor(2εn) + 1) + 1` methods, whatever the analysis. Our classifier beating the
architecture-only rules (gap 0.140) needs labels better than 7.0% error and stands. The
profile HMM beating our classifier (gap 0.039) needs better than 1.9% and does **not**.
Bound and Lean 4 proof from DisorderNet (T. Marena, unpublished).

---

## 11. Candidate models

**Choice.** `experimental/` holds challengers that replace nothing unless they win a
measurement. Production code cannot import a candidate, enforced by parsing the imports of
every production module rather than by convention.

**Promotion requires all three**: macro-F1 above the incumbent, the incumbent falling
outside the challenger's confidence interval, and beating the profile HMM. Two of three is
not promotion.

**Why the grammar challenger reads no annotation.** Its score is then what would be
available for a protein nothing has annotated — which is the actual bottleneck for
non-model organisms. A test greps its source for annotation-derived fields to keep that
true.

---

## 12. Cluster practice

- Stages chained with `--dependency=afterok`, so SLURM waits and nothing polls a login node.
- BLAS threads pinned to 1 per worker. Unpinned, workers ran at ~189% CPU each — roughly
  twice the cores allocated, degrading every other job on a shared node.
- More chunks than workers, so a slow chunk cannot strand a worker and progress is visible
  before the run ends.
- MAFFT is vendored as a portable binary: Rockfish exposes it only through licensed
  SBGrid/BioGrids module trees that refuse to load in a batch job.
