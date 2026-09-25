# A deep classifier for JDPs: design, and what has to be true for it to work

This is a design review of the proposed hierarchical network, written against what the
rest of this repository has already measured. The architecture is sound in outline. The
parts most likely to decide whether it works are not in the architecture at all, so they
are treated first.

## 1. The three problems that decide the outcome

### 1.1 Label circularity

The proposal trains on "all UniProt proteins matching PF00226, using the broad noisy
classifications as weak labels". Those classifications come from the same domain
architecture the network is meant to transcend. A model trained on them learns to
reproduce a rule system, and its agreement with that system is then reported as accuracy.
It would score well and mean nothing.

The labels have to come from somewhere the architecture does not:

- **Curated nomenclature.** Reviewed UniProt entries named `DnaJ homolog subfamily A/B/C
  member N`. `validation/labels.py` already extracts these; the full set is **129
  proteins** (A: 15, B: 52, C: 62). Small, but genuinely independent of the rules.
- **Structure.** AlphaFold or PDB models give domain boundaries and fold assignments that
  do not depend on sequence-signature matching.
- **Function.** Verified Hsp70 partner specificity, complementation data, localisation.

129 labels will not train a 3B-parameter model. That is the actual constraint, and it
points the design toward a frozen backbone with a small head — which the proposal already
has — and away from anything with more capacity.

### 1.2 Evaluation split

**This is the single most important decision in the plan.** A random train/test split over
a JDP set leaks: UniProt contains many near-identical JDPs from related strains, so a
random split puts homologs of the test proteins in training. Reported accuracy will be
high and will not survive contact with a new organism, which is exactly the use case.

Split by **taxonomic clade**, holding out whole phyla, and additionally cluster at ≤30%
sequence identity (MMseqs2 or CD-HIT) and keep clusters intact across the split. Report
both a random-split and a clade-held-out number. The gap between them *is* the result —
it measures how much of the model's performance is memorised homology.

The repository's own profile-HMM baseline (`validation/homology.py`) documents this same
limitation for its cross-validation rather than hiding it; the network should be held to
the same standard.

### 1.3 The baseline that must be beaten

Before any of this is justified, the network must beat:

| Baseline | Where |
|---|---|
| Majority class (predict C) | `validation/metrics.py` |
| Architecture rules alone | `jdp_classifier/rules.py` |
| Profile HMM, cross-validated | `validation/homology.py` |
| Grammar features + naive Bayes | `experimental/grammar_classifier.py` |

The last one matters most for deciding what to build next. It uses no annotation and no
learned representation. If a linear model over twenty compression and entropy features
comes close to the network, the gain was in the features, not the capacity, and the
network is the wrong investment.

## 2. Architecture, revised

### 2.1 Backbone

Start with **ESM-2 `t33_650M`**, not `t36_3B`. On domain-level classification the 650M
checkpoint is typically within a point or two of the 3B one, at a fifth of the memory, and
fits alongside a batch on a single A100 — which is what Rockfish provides. Move up only
once the 650M model is demonstrably capacity-limited rather than data-limited. With 129
gold labels it will be data-limited.

Keep the backbone frozen initially and **cache the embeddings to disk**. Residue embeddings
for 181,526 proteins are computed once and reused across every experiment; recomputing
them per epoch would dominate the entire training budget.

LoRA on the last four layers' query and value projections is reasonable, but add it only
after the frozen-backbone baseline is measured. It is a second variable, and introducing it
alongside everything else makes a disappointing result uninterpretable.

### 2.2 Segmentation

The proposed CNN + BiLSTM soft tokenizer is the weakest-justified component, because a
non-learned segmentation already exists and works: `domain_layout/regions.py` routes every
residue using InterPro boundaries and metapredict disorder, and `domain_layout/msa.py` and
`shark.py` consume the result.

Use that as **supervision** rather than replacing it. Train the boundary head against the
existing routing as a target on the 181k proteins where InterPro annotates them, then let
it run free on proteins where InterPro is silent. That converts an unsupervised problem
into a supervised one with 181,526 training examples, and the segmentation stays
interpretable — a predicted boundary can be compared against a known one.

A boundary-F1 metric against held-out InterPro boundaries then tells you whether the
tokenizer works, independently of whether the classifier does. Without that, a poor final
score cannot be attributed.

### 2.3 Syntax transformer

Relative position encodings over domain tokens are the right call, and 4–6 layers is
right for the data volume. Two additions:

- **Encode linker length explicitly**, bucketed on a log scale. Spacing between a J-domain
  and its partner is functionally meaningful, and a continuous distance term buried in
  attention is harder to inspect than an explicit feature.
- **Attach the grammar vector** from `domain_layout/grammar.py` to each token. Entropy
  rates and shuffle-controlled order scores are cheap, and they describe exactly the
  low-complexity linkers where a pLM's residue embeddings are least informative.

### 2.4 Heads

Cross-entropy plus contrastive is right, with one correction. If contrastive positives are
"same class", the contrastive head re-learns the labels and adds nothing beyond a
regulariser. Draw positives from **augmentations of the same protein** — BLOSUM62
substitution jitter, terminal truncation, linker-length jitter — so the objective learns
invariance to the transformations that genuinely preserve function.

For uncertainty, prefer an **evidential / Dirichlet** head over MC dropout. MC dropout
measures sensitivity to its own noise, which correlates only loosely with being
out-of-distribution; a Dirichlet head gives explicit mass to "none of the above", which is
the actual question for a JDP from an unstudied organism.

Calibrate it. An uncertainty head that is never checked against held-out clades is
decoration — measure expected calibration error, and check that held-out phyla receive
genuinely higher uncertainty than held-out members of trained phyla.

## 3. Nomenclature

The proposal says "Classes I–IV". This project uses **A / B / C**, which maps onto the
older I / II / III. There is no consensus class IV. Whatever is chosen, it must be stated
explicitly and matched to `data/reference_jdps/reference_classes.json`, or the model's
outputs cannot be compared to anything else here.

## 4. Order of work

Each step is cheap relative to the one after it, and each can fail informatively:

1. **Cluster and split by clade.** Nothing measured before this exists is trustworthy.
2. **Run the existing baselines on that split.** They are already implemented.
3. **Cache ESM-2 650M embeddings**, mean-pooled per routed region. Fit the same linear
   classifier on those instead of grammar features. This isolates the value of the
   representation with no architecture at all.
4. **Add the syntax transformer** over region embeddings. This isolates the value of
   modelling layout.
5. **Add the learned tokenizer**, supervised by existing routing. This isolates the value
   of not needing InterPro.
6. **Add LoRA, contrastive, and uncertainty** heads, one at a time.

If step 3 does not beat the profile HMM, steps 4–6 are unlikely to rescue it, and the
finding — that a 650M-parameter protein language model does not beat `hmmsearch` on JDP
classification — is worth reporting on its own.

## 5. Honest expectations

Class A/B/C is largely *defined* by domain architecture. A model that reads architecture
well will approach the ceiling of what the labels contain, and the incumbent rule system
already reads architecture directly. The plausible gain is not on the 129 curated
proteins; it is on the proteins InterPro cannot annotate at all, where the rules have
nothing to work with and a sequence-only model still does.

That is the experiment worth designing for, and it needs a held-out set of JDPs with
*known* class and *absent* InterPro annotation. Building that set is the prerequisite for
the whole plan, and it is not yet built.
