# Completed GB.DNA-300M contrastive fine-tuning: v002

Training completed on **2026-10-01 at 07:19 UTC** (00:19 Pacific). One epoch
processed **25,000 triplets in 12,500 microbatches and 1,563 optimizer updates**.
The final model is `results/finetuning/gbdna300m_exon_infonce_v002/training/checkpoint-001563/`.
This report covers training only. Fine-tuned reference-vector and WES retrieval
results are reported separately in the [completed reanalysis](finetuned_v002_reanalysis_results.md).

## Hypothesis and design

The hypothesis is that contrastive training can bring a short fragment closer to
its source gene's representation, improving retrieval over the unmodified
foundation model. This is task-specific fine-tuning, not a zero-shot result.

- 250 training genes, 100 triplets per gene; 25 additional validation genes,
  100 prepared triplets per gene. All 12 WES target genes are excluded from
  training anchors, negative donors and validation.
- Each triplet comprises a full selected CDS anchor, one same-gene positive
  fragment and one different-gene negative. Both fragments are 50–250 bp where
  exon length permits and fit within one selected CDS block. The audit found
  zero junction crossings across all 27,500 prepared triplets.
- Shared full-parameter GB.DNA-300M encoder, float32 SDPA on MPS, mean pooling
  within 3,998-base contexts and length-weighted pooling for whole-CDS anchors.
- Two triplets per microbatch; eight accumulated backward passes per optimizer
  update (normally 16 triplets). Negatives are shared only within a microbatch,
  not across the gradient-accumulation window. Known false negatives are masked.
- Cosine InfoNCE cross entropy, temperature 0.05; AdamW learning rate 1e-5,
  weight decay 0.01, gradient-norm clipping at 1.0; seed 42.

See [the detailed methods and recovery record](contrastive_finetuning_v002.md)
for sampling, checkpoint configuration and reproducible commands.

## Training and validation results

Mean training loss fell from **1.013** over the first 50 updates to **0.241** over
the final 50. Pre-clipping gradient norm exceeded 1.0 on **99.55%** of updates;
clipping was therefore active almost throughout the run, rather than a rare
intervention. This does not alone establish instability or successful learning.

| Optimizer update | Fixed-subset InfoNCE loss | Within-batch accuracy |
|---|---:|---:|
| 250 | 0.741 | 68% |
| 500 | 0.586 | 76% |
| 750 | 0.680 | 68% |
| 1,000 | 0.743 | 72% |
| 1,250 | 0.536 | 76% |
| 1,500 | 0.734 | 64% |
| 1,563 | 0.738 | 72% |

Validation reuses one fixed triplet for each of the 25 validation genes, in
12 two-triplet batches and one singleton batch. Each loss point is the unweighted
mean of 13 batch losses; accuracy is total correct positive-fragment selections
divided by 25. The singleton has a smaller candidate set and contributes more
loss weight per triplet. Evaluation runs without gradients or parameter updates.

The lowest sampled validation loss was **0.536 at update 1,250**; final loss was
**0.738**, with **18/25 correct selections (72%)**. Validation fluctuated while
training loss declined. This is consistent with a generalization gap, but the
small diagnostic subset does not establish its cause. There is no pre-training
validation baseline, no exhaustive 2,500-triplet validation, and no WES mapping
accuracy in these training diagnostics. We retained the final checkpoint;
we did not select the best-validation weights, and checkpoint 1,250 was pruned
under the two-checkpoint retention policy.

## Execution and memory recovery

Two attempts stopped with MPS allocation-limit errors, after updates 469 and
351. They were resumed from checkpoint 250; losses at matching updates reproduced
the same trajectory to numerical precision. Discarded updates are not additional
successful training epochs and are not concatenated into the final learning curve.

Context-level activation checkpointing, explicit collection/cache release after
backward, and release of temporary checkpoint-loading references were added.
The stable recovery used checkpoints every 25 updates, validation every 250,
and the unchanged MPS memory fraction of 0.6. It reached its 13.5-hour budget at
update 1,490; the authorized watcher resumed it and completed the epoch. No
additional out-of-memory failure was recorded in that recovery/continuation.
The stable recovery plus final continuation took about **14 hours 23 minutes**
of wall time, excluding the earlier failed attempts and pilots.

The 10,500 post-backward memory samples cover recovery from update 250 onward.
Maximum sampled driver allocation before cleanup was **14.81 GB**; maximum
sampled live tensor storage was **4.83 GB**. These are decimal GB and discrete
samples, not within-operation peaks. RSS and device allocations overlap unified
memory and should not be summed. The logs support substantial reclaimable cache
pressure, but do not conclusively reconstruct the earlier uninstrumented failures.

## Artifacts and diagnostics

```text
results/finetuning/gbdna300m_exon_infonce_v002/
├── data/                         # Gene splits, triplets, audit, checksums
├── pilot_mps_b2/                 # Feasibility measurements; discarded weights
├── training/
│   ├── checkpoint-001563/        # Final weights, tokenizer, optimizer/RNG state
│   ├── training_manifest.json   # status: complete
│   ├── training_log.jsonl       # Completed trajectory, one row per update
│   ├── validation_log.jsonl     # Validation and skipped-validation records
│   ├── memory_log.jsonl         # Before/after cleanup measurements
│   ├── attempt_01_failed/       # Preserved failed-trajectory logs
│   ├── attempt_02_failed/
│   └── diagnostics/             # PNG/PDF figures, CSV metrics, summary/provenance
└── budget_resume_status.json    # Authorized automatic continuation: complete
```

[Open the local plot report](../results/finetuning/gbdna300m_exon_infonce_v002/training/diagnostics/README.md).
It includes learning curves, validation accuracy, gradient norms, memory cleanup,
runtime versus input length, configured learning rate and attempt comparisons.
Results files may be absent from public clones. A compact
[completed-run metrics snapshot](benchmarks/contrastive_gbdna300m_v002_completed.json)
is retained in documentation. Generate figures with
`scripts/report_training_diagnostics.py` and the experiment directory.

Final weight SHA-256:
`0fe34bd24bc666be421dad9dea0a92cb76a3482f82a4c582b4feb1ee6e6641de`.

The [downstream reanalysis](finetuned_v002_reanalysis.md) uses these weights for
both reference and query vectors, with a separate reference-only UMAP transform.
Its completed results are documented separately to distinguish training from retrieval evaluation.
