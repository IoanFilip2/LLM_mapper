# Contrastive gene–substring fine-tuning: v001

The smaller [v002 exon-contained plan](contrastive_finetuning_v002.md) supersedes
v001 as the proposed full-run configuration. v001 is retained as a feasibility baseline.

## Status and hypothesis

**The dataset and training feature are implemented; only a bounded MPS pilot has
run. No full training run has started and no production fine-tuned checkpoint
exists.** The 12 pilot updates were discarded. The base model, its reference
vectors, and the completed zero-shot experiment remain the baseline.

The hypothesis is that explicitly attracting a short fragment to its parent gene
while repelling unrelated fragments will improve read-to-gene cosine retrieval.
The initial GB.DNA-300M study had 0/935 correct top-1 calls among all HGNC genes,
despite every query being an exact substring of its reference CDS. Contrastive
fine-tuning directly targets that mismatch between short-read and pooled-CDS
representations. Improvement remains an experimental question, not a pilot result.

[Holur et al., Embed-Search-Align (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11919449/)
provides the inspiration: a shared sequence encoder learns relative read/reference
similarity through contrastive self-supervision, followed by retrieval and local
alignment. Here we fine-tune GB.DNA-300M using whole selected CDS anchors and
50–250-bp substrings. This is our own experiment, not a reproduction of that
paper's encoder architecture, reference segmentation, or alignment performance.

## Dataset and held-out evaluation

The immutable dataset is under
`results/finetuning/gbdna300m_cds_infonce_v001/data/`.

| Split | Genes | Positive pairs | Explicit negative pairs |
|---|---:|---:|---:|
| Training | 1,000 | 200,000 | 200,000 |
| Validation | 100 separate genes | 2,000 | 2,000 |
| WES test | 12 existing target genes | Existing 935 reads | Not used in training |

The user-approved 12-gene holdout is TP53, KRAS, HLA-A, HLA-B, HLA-C, EGFR, MYC,
BRCA1, BRCA2, CDKN2A, BRAF and MDM2. These genes cannot be training anchors,
negative donors, or validation genes. Training and validation genes are also
disjoint. This is a **gene-level holdout**, not a gene-family/homology-disjoint
split; related sequences can occur across genes, and pretraining exposure is not
controlled by this split.

The source is the same HGNC-selected **spliced CDS FASTA** used in the reference
database, not full intron-containing genomic gene spans. The source hash and
base-model fingerprint are checked before dataset creation. Seed 42 selects
1,100 eligible genes uniformly without replacement; the first 1,000 are training,
and the remaining 100 validation. Genes shorter than 50 bases are excluded.

For each anchor, sample 200 positive and 200 negative pairs with replacement:

1. Draw a positive length uniformly from integer lengths 50 through
   `min(250, anchor CDS length)`, and a valid start uniformly within that CDS.
2. Choose a different donor gene uniformly from the **same split**, then draw its
   substring length and start similarly. Reject a negative if it or its reverse
   complement occurs exactly within the anchor. Near-homologous negatives may
   still occur; this is not a full homology screen.
3. Independently reverse-complement each fragment with probability 0.5 to train
   strand robustness. The full gene anchor keeps transcript orientation.

The training genes range from 75 to 43,521 bp, with median 1,297.5 bp. Their
reference representations require 1,094 contexts at the 3,998-base payload limit.
All 202,000 training-plus-validation triplets passed bounds, different-donor,
exact-negative exclusion, split, and holdout audits. Compact integer offsets
plus one copy of each gene sequence use approximately **8.5 MB**, rather than
storing full duplicated gene sequences in 400,000 pair records.

**WES caveat:** this literal random-CDS-substring design permits exon joins.
The audit found **112,233/200,000 (56.12%) positive training substrings cross a CDS
junction** and therefore need not be contiguous genomic WES reads. The current
v001 plan preserves the requested random-substring experiment. Before choosing
a full WES-focused run, consider a separately versioned dataset restricting
substrings to genomic exon/CDS blocks. The completed WES evaluation itself uses
reads contained in single CDS blocks. No such dataset change was silently applied.

## Siamese encoder and InfoNCE

One shared, trainable encoder processes the gene and both fragments. Rather than
encode the same gene twice for its positive and negative pair, the implementation
uses one triplet `(gene, positive fragment, negative fragment)`. Thus 400,000
labeled pairs correspond to **200,000 triplets per epoch**. No cached base-model
gene embeddings are used as fixed training targets: both branches receive
gradients through the shared encoder.

For microbatch size B, the fragment candidate list contains B positives followed
by B explicit negatives. For gene i:

```text
z_i = L2_normalize(encoder(gene_i))
v_j = L2_normalize(encoder(fragment_j))
s_ij = cosine(z_i, v_j) / temperature
loss_i = -log(exp(s_i,i) / sum(exp(s_i,j) for allowed candidates j))
loss = mean(loss_i)
```

The paired positive is retained. Other fragments from the same gene, or fragments
that occur exactly in that anchor in either orientation, are masked from its
negative set. Every anchor must retain at least one negative. This is **gene-to-
fragment InfoNCE**, not binary cross-entropy or a fitted gene-label classifier.
The denominator uses explicit plus in-batch negatives; with B=2 there are at most
**three negatives per anchor**. Eight gradient-accumulation steps produce an
optimizer batch of 16 triplets, but **do not increase the InfoNCE negative pool**.
A larger contrastive batch or a carefully designed queue is a separate choice.

The full 299,212,800-parameter encoder is trainable. No projection head or LoRA
adapter is added. Model/tokenizer revision and pooling match the base reference.
For genes longer than 3,998 bases, every nonoverlapping context is encoded and
its raw pooled vector weighted by base length. The resulting gene vector is
normalized once. Gradients flow through all contexts; long genes are not silently
truncated. Fragments use the same nucleotide mean pooling and normalization.

The former inference-only GB SDPA adapter now supports backpropagation and uses
the original attention dropout probability in training. The base checkpoint's
dropout probabilities remain zero; v001 does not add the paper's dropout/noise
recipe. Forward and gradient parity against eager attention are tested on a tiny
RNABert, including gradient checkpointing and multi-context pooling.

## Proposed full-run configuration

The versioned configuration is
[`configs/finetuning/gbdna300m_cds_infonce_v001.json`](../configs/finetuning/gbdna300m_cds_infonce_v001.json).

| Setting | Value |
|---|---|
| Parent | GB.DNA-300M revision `a833c5602b0dad0728c0aeeecaf43086ebeb96ca` |
| Precision / attention | float32 / SDPA |
| Trainable parameters | 299,212,800 |
| Optimizer | AdamW, constant learning rate 1e-5, weight decay 0.01 |
| Temperature | 0.05 |
| Microbatch | 2 triplets |
| Gradient accumulation | 8 microbatches, normally 16 triplets/update |
| Gradient clipping | Global norm 1.0; reject nonfinite loss/gradients |
| Gradient checkpointing | Enabled, non-reentrant |
| Initial proposed epochs | 1 |
| Steps per epoch | 100,000 microsteps; 12,500 optimizer updates |
| Checkpoint / validation interval | Every 250 optimizer updates and epoch end |
| Complete checkpoint retention | Newest two, within this experiment only |
| MPS memory cap | 60% of the Metal recommended maximum, not 60% of physical RAM |

Validation uses a fixed, gene-balanced subset (one triplet per validation gene
at these defaults) and reports InfoNCE and within-batch retrieval. That diagnostic
is **not** whole-reference mapping accuracy. The 12 WES test genes are excluded
from checkpoint validation/model selection. No improvement claims are made from
losses recorded during the timing pilot.

## Measured feasibility on this Mac

The [retained pilot and dataset audit](benchmarks/contrastive_gbdna300m_v001.json)
contains the individual measurements and source fingerprints.

The pilot performed one warmup, two training microsteps in each of five equal-count
gene-length strata, and one worst-length step: **12 updates total**. Each measured
step included forward, backward, gradient clipping, and AdamW, so optimizer states
were allocated. Two copies of the longest selected CDS were used in the stress
microbatch, covering a conservative length combination.

| Measurement | Result |
|---|---:|
| Physical unified memory | 51.54 GB (48 GiB) |
| Peak sampled MPS driver allocation | **21.94 GB** (20.44 GiB) |
| Peak sampled process RSS | 1.25 GB; overlaps device memory, **do not add** |
| Parameter + gradient + Adam-state analytical floor | 4.79 GB |
| Mean forward/backward per 2-triplet microbatch | 3.837 s |
| Mean optimizer step | 0.404 s |
| Longest-CDS stress forward/backward | 110.15 s |
| Projected training compute per epoch | **107.98 hours (~4.5 days)** |
| Coarse planning range | **75.6–162.0 hours (~3.2–6.7 days)** |

The estimate is `100,000 × mean forward/backward time + 12,500 × mean optimizer
time`. Pilot AdamW runs after every microstep to test memory allocation; the full
run would accumulate eight before an optimizer update. The length-stratified
sample is small, and the extreme stress step is excluded from the population
mean estimate. The range is a rough planning allowance, not a confidence interval.
Validation, checkpoint I/O, initial loading, thermal effects, and changing machine
load are extra. Inference throughput is not substituted for training throughput.

The run **fits this machine's tested memory budget**, but the full experiment is
slow enough that a smaller learning-curve run or a CUDA training machine merits
consideration before committing to several days. No larger microbatch has been
benchmarked and no speedup from it is assumed. More genes/epochs scale the work;
more accumulated gradients do not remove the repeated gene forward/backward cost.

A complete full-parameter checkpoint is approximately **3.6 GB** (weights plus
Adam states). Retaining two uses about 7.2 GB, with about 10.8 GB temporarily while
writing the next one. Checkpoint serialization may require extra host memory;
that peak was not measured by the pilot. Only this experiment's old, complete
checkpoints are pruned. Interrupted `.incomplete-*` directories are not accepted
as checkpoints and are left for inspection. Base-model cache files are never
modified or pruned by this feature.

## Versioning and subsequent evaluation

```text
configs/finetuning/gbdna300m_cds_infonce_v001.json
results/finetuning/gbdna300m_cds_infonce_v001/
  experiment.json            # Prepared/piloted status; full training not started
  data/                      # Immutable gene splits, triplet offsets, hashes and audit
  pilot_mps_b2/              # Timings, memory, status; no retained model weights
  training/                  # Created only by an explicit future train command
    training_manifest.json
    training_log.jsonl
    latest.json
    checkpoint-NNNNNN/
      model.safetensors      # Canonical HF weight names; shared inference API reloads it
      config.json, vocab.txt
      inference_config.json  # Model label/config for this checkpoint, separate from base
      checkpoint_manifest.json
      training_state.pt      # Optimizer, RNGs, exact epoch/microstep position
```

Each checkpoint has input/config/model fingerprints and an atomic completion
rename. Resume validates checksums and experiment settings, restores optimizer
and RNG state, and resumes only at an optimizer boundary. Synthetic tests confirm
that interrupted/resumed training matches uninterrupted training exactly. State
files are trusted local pickle artifacts; do not load untrusted checkpoints.
Use a new experiment identifier (`v002`, etc.) for changed data or training
settings. Existing datasets/checkpoint directories are not overwritten.

After selecting a trained checkpoint, **re-embed all reference genes and all WES
queries with that checkpoint** into a new results directory. Base-model vectors
must not be mixed with fine-tuned query vectors. Fit a new **reference-only** UMAP
for that new model, save its transform, and project its queries through that
transform. Keep the current base-model UMAP/results as their own fixed baseline;
its numerical space is not the fine-tuned model's space. Compare the same held-out
935 reads, both all-HGNC and 12-target metrics, including the Q30 subset.

## Commands from the project root

Dataset preparation and a bounded pilot are separate from full training:

```bash
.venv/bin/python -m scripts.finetune_contrastive prepare \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  results/finetuning/gbdna300m_cds_infonce_v001/data \
  --train-genes 1000 --validation-genes 100 --pairs-per-gene 200 \
  --validation-pairs 20 --min-length 50 --max-length 250 --seed 42

HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.finetune_contrastive pilot \
  results/finetuning/gbdna300m_cds_infonce_v001/data \
  results/finetuning/gbdna300m_cds_infonce_v001/pilot_mps_b2 \
  --device mps --microbatch 2 --repeats 2
```

Those output directories already exist after this preparation; use new versioned
paths to generate a different dataset or repeat the pilot. The following is
**documented for a future approved full run and has not been executed**:

```bash
HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.finetune_contrastive train \
  results/finetuning/gbdna300m_cds_infonce_v001/data \
  results/finetuning/gbdna300m_cds_infonce_v001/training \
  --config configs/finetuning/gbdna300m_cds_infonce_v001.json --device mps
```

To resume, add `--resume` followed by a complete checkpoint directory under the
same experiment and keep the same configuration. Training is never invoked by
`prepare` or `pilot`.

## Verification

**73 tests passed, one existing test skipped.** New coverage includes reproducible
pair sampling, InfoNCE probabilities and two-branch gradients, false-negative
masking, eager/SDPA training gradients, checkpointed context pooling, canonical
checkpoint export/reload, exact resume behavior, and checkpoint retention. All
400,000 training pairs and 4,000 validation pairs were audited. The actual MPS
pilot completed with finite losses/gradients and no out-of-memory failure.
