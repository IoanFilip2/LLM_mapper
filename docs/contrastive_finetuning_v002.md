# v002: smaller contrastive run with exon-contained fragments

**Status: training complete; all 1,563 updates finished on 2026-10-01 at 07:19 UTC.** This supersedes
v001 as the current experiment. The base model and v001 artifacts remain
separate. See the [completed-run results](contrastive_finetuning_v002_results.md)
for the final metrics and interpretation. The sections below retain the original
plan and chronological recovery record.

## Dataset

| Split | Genes | Triplets per gene | Positive pairs | Negative pairs |
|---|---:|---:|---:|---:|
| Training | 250 | 100 | 25,000 | 25,000 |
| Validation | 25 additional genes | 100 | 2,500 | 2,500 |

“100 contrastive pairs” is implemented consistently with v001 as 100 positive/
negative **triplets** per gene, rather than 100 binary-labeled examples total.
Each triplet encodes the gene once with one positive and one negative fragment.
This is one-eighth of v001's training triplets. Seed 42 draws a fresh reproducible
split from eligible HGNC genes. The 12 WES target genes remain excluded from
training anchors, negative donors, and validation; donors stay within their split.

**Both positive and negative fragments must fit entirely inside one selected
transcript CDS block.** First draw a length uniformly from 50 through the smaller
of 250 and the longest CDS-block length. Then draw uniformly among all valid
start positions across blocks that accommodate that length. Blocks are therefore
weighted by their number of valid starts at the chosen length. Genes lacking a
50-base CDS block are excluded. Reverse-complement augmentation and exact-match
negative rejection are unchanged from v001.

The block file's transcript identifiers, CDS bounds, genomic/CDS length agreement,
and complete CDS tiling are checked against the reference gene catalog. Its SHA-256
is retained with the dataset. An exhaustive audit verified **zero junction-crossing
positive or negative fragments** across all 27,500 triplets, along with per-gene
counts and split/holdout restrictions. All data are sliced from the same CDS
FASTA as the base reference database.

The **gene anchors still represent complete spliced CDS**, including length-weighted
pooling over all contexts for long genes. This change makes the sampled fragments
WES-compatible; it does not change the reference representation to exon windows.
The original held-out WES cohort and its labels remain unchanged.

## Configuration and measured feasibility

The shared full-parameter encoder, cosine InfoNCE, temperature 0.05, float32 SDPA,
AdamW learning rate 1e-5, and gradient checkpointing follow
[the v001 method](contrastive_finetuning.md). Microbatch size is two triplets with
eight accumulation steps. One epoch has **12,500 microsteps and 1,563 optimizer
updates**; the final optimizer batch is smaller. Gradient accumulation does not
increase the in-batch negative pool.

A new 12-update MPS pilot used this actual dataset, including a stress batch with
two 43,521-base anchors. Pilot weights were discarded.

| Measurement | Result |
|---|---:|
| Mean forward/backward per microbatch | 3.247 seconds |
| Mean optimizer update | 0.396 seconds |
| Projected training compute per epoch | **11.45 hours** |
| Coarse compute-only planning range | **8.0–17.2 hours** |
| Peak sampled MPS allocation | **21.99 GB** |
| Peak sampled process RSS | 1.28 GB; do not add to overlapping MPS memory |
| Physical unified memory | 51.54 GB (48 GiB) |

The central estimate is compatible with a **12–14-hour planning window**, but the
small pilot cannot guarantee epoch completion within that time. Validation,
checkpoint I/O, machine load and thermal effects are extra; the estimate excludes
the longest-gene stress step from its population mean. The range is a planning
allowance, not a statistical confidence interval.

The [v002 configuration](../configs/finetuning/gbdna300m_exon_infonce_v002.json)
sets **`max_run_hours: 13.5`**, leaving nominal headroom within the accepted 14-hour
window. The runner checks this budget after optimizer updates, saves a resumable
checkpoint, skips further validation at that stop, and reports `budget_exhausted`
if the epoch is unfinished. It does **not** label a partial epoch complete. A
current accumulation window and checkpoint I/O can extend elapsed time past the
13.5-hour threshold; this is a graceful stop budget, not a hard process kill or
guaranteed 14-hour deadline. A resumed invocation gets its own runtime budget.

Validation retains 2,500 triplets. As in v001, periodic validation uses one fixed
sampled triplet per validation gene (25 at these defaults) for a quick diagnostic;
this is not exhaustive validation or whole-reference mapping accuracy. The 12 WES
genes remain reserved for the later before/after retrieval comparison.

## Artifacts and commands

- Dataset/pilot: `results/finetuning/gbdna300m_exon_infonce_v002/`
- [Retained dataset audit, pilot and configuration](benchmarks/contrastive_gbdna300m_v002.json)
- [Versioned training configuration](../configs/finetuning/gbdna300m_exon_infonce_v002.json)
- Training checkpoints: `results/finetuning/gbdna300m_exon_infonce_v002/training/`

From the project root, the preparation/pilot commands used are:

```bash
.venv/bin/python -m scripts.finetune_contrastive prepare \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  results/finetuning/gbdna300m_exon_infonce_v002/data \
  --train-genes 250 --validation-genes 25 --pairs-per-gene 100 \
  --validation-pairs 100 --min-length 50 --max-length 250 --seed 42 \
  --cds-blocks data/reference/GRCh38-ensembl-110/hgnc_cds/cds_blocks.jsonl

HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.finetune_contrastive pilot \
  results/finetuning/gbdna300m_exon_infonce_v002/data \
  results/finetuning/gbdna300m_exon_infonce_v002/pilot_mps_b2 \
  --device mps --microbatch 2 --repeats 2
```

These immutable output directories now exist; use another version for another
preparation/pilot. **The full training command below was launched:**

```bash
HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.finetune_contrastive train \
  results/finetuning/gbdna300m_exon_infonce_v002/data \
  results/finetuning/gbdna300m_exon_infonce_v002/training \
  --config configs/finetuning/gbdna300m_exon_infonce_v002.json --device mps
```

Checkpoint naming, newest-two retention, checksum validation, and resume follow
v001. After training, reference **and** query embeddings must be rebuilt together
under the selected checkpoint, with a new reference-only UMAP and saved transform.

**Verification:** 75 tests passed, one existing test skipped. New tests cover
exon boundary sampling for both fragment types, dataset block validation and
split integrity, and budget-stop/resume behavior without falsely marking an
unfinished epoch complete.

## Monitoring the launched run

The run is detached from the terminal and uses `caffeinate -i` to prevent idle
system sleep while active. Keep the Mac powered and awake; this does not protect
against shutdown or closing the lid. Launch details are saved in `launch.json`
at the experiment root. To inspect progress from the project root:

```bash
tail -n 5 results/finetuning/gbdna300m_exon_infonce_v002/training/training_log.jsonl
cat results/finetuning/gbdna300m_exon_infonce_v002/training/training_manifest.json
```

The log records loss and gradient norm after each optimizer update (1,563 total
for the epoch). The manifest reports `complete` only after the epoch finishes,
or `budget_exhausted` after a resumable budget stop. A stale `training` status
alone does not establish that the process is still alive; also inspect
`training_console.log` and the process ID recorded in `launch.json` if progress
stops. Checkpoints are saved every 250 updates and at completion/budget stop.

### Persistent training metrics

`training/training_log.jsonl` is flushed after every optimizer update and retains
epoch, microstep, optimizer step, mean training InfoNCE across the accumulation
window, and pre-clipping gradient norm. Individual microbatch losses are not
saved separately in this run.

For the active v002 run, a lightweight observer archives every completed
checkpoint manifest into `training/metrics_history/checkpoint-NNNNNN.json`.
These small records survive pruning of model checkpoints and include validation
InfoNCE, within-batch contrastive accuracy, sample count, configuration and step.
The observer polls every 15 seconds; launch details and errors are stored in
`metrics_observer.json` and `metrics_observer.log` at the experiment root.
Validation occurs every 250 optimizer updates and at epoch completion; a budget
stop skips validation and records that fact. These diagnostics use the fixed
25-triplet validation subset, not all 2,500 validation triplets or WES accuracy.

The runner also now writes `training/validation_log.jsonl` directly for future
invocations/resumes, including elapsed seconds. This code change does not alter
the already-running process; its validation history is preserved by the observer.

## Recovery from the first MPS memory failure

The first invocation stopped after update 469 (microstep 3752), with an MPS
allocation-limit error during the next accumulation window. That window contains
a 43,521-base anchor at microstep 3754. The last durable checkpoint was update
250. Training loss averaged 1.013 across the first 50 updates and 0.459 across
the last 50 before failure; checkpoint-250 validation loss was 0.741 with 68%
within-batch accuracy on the 25-triplet diagnostic subset. These are not WES
retrieval results.

Recovery adds non-reentrant activation checkpointing around each complete
context encoding, in addition to per-layer checkpointing. This avoids retaining
all contexts' internal activations for a long CDS until backward. Contexts are
recomputed during backward; no sequence is truncated and the model, sampling,
loss, microbatch, accumulation and 0.6 MPS memory fraction stay unchanged. Tiny
model tests compare parameter gradients with checkpointing disabled. Recomputing
contexts can increase runtime, so the original pilot estimate may no longer apply.

The bounded recovery probe (`scripts/verify_training_memory.py`) loads the saved
weights and optimizer, tests the eight-microbatch window after update 469, then
a batch containing two longest-gene anchors. Its weights are discarded. The
report is `memory_recovery_probe.json` in the experiment directory. This probes
peak workloads but cannot guarantee absence of memory growth in a long run.

A resumed invocation repeats updates after 250. The original training log is
retained separately so metrics from the abandoned trajectory are not mixed with
the resumed trajectory. Future exceptions now set manifest status to `failed`
and retain the last durable checkpoint pointer. The resumed invocation receives
a fresh 13.5-hour graceful-stop budget.

Recovery verification passed: the eight-microbatch failure window and the
two-longest-anchor stress batch completed with finite loss and gradients.
Post-microbatch driver-memory samples reached 13.73 GB; these are discrete
samples, not a continuously measured peak. The combined probe took 247.9 seconds.
All 75 tests passed, with one existing skip.

The resumed process completed update 251 (loss 0.371, gradient norm 5.439).
Launch metadata is in `resume_02.json`, console output in `training_resume_02.log`,
and the original failed trajectory in `training/attempt_01_failed/`. The primary
training log keeps updates 1–250 followed by the resumed trajectory. Validation
now also appends directly to `training/validation_log.jsonl`.

## Sustained memory profiling and more frequent checkpoints (attempt 03)

Attempt 02 failed at update 351 despite passing the short stress probe. Its
metrics and manifest are preserved in `training/attempt_02_failed/`. This shows
that the short probe was insufficient to establish long-run memory stability.

The [recovery configuration](../configs/finetuning/gbdna300m_exon_infonce_v002_memory_recovery.json)
now saves checkpoints every **25 optimizer updates** (next: 275) and keeps the
newest two. Validation remains every **250 updates** and at epoch completion.
Resume compatibility permits changes only to operational settings such as these
intervals, logging, cleanup and runtime budget; dataset, device, learning rate,
microbatch and other scientific settings must still match. Both original
configuration and launch metadata are preserved.

The sustained profile runs as part of resumed training, retaining useful updates.
After every microbatch, `training/memory_log.jsonl` records elapsed seconds,
anchor lengths, live MPS tensor bytes, total MPS driver bytes and process RSS:
first before cleanup, then after Python garbage collection, then after MPS
cache release. This distinguishes reclaimed tensor storage from unused cached
allocations. Driver-minus-tensor memory also includes backend allocations, so
it must not be interpreted entirely as reclaimable cache. RSS overlaps unified
GPU memory; do not sum these fields. Samples do not measure within-forward peaks.

Explicit collection/cache release occurs after each backward pass, with gradients
retained for accumulation. CPU resume-state references are also released after
loading. These interventions preserve the loss and optimizer schedule but can
slow execution. They are being evaluated, not asserted to solve the failure.
The same MPS allocation limit and fresh 13.5-hour stop budget apply.

Launch: `resume_03.json`; console: `training_resume_03.log`.
Training loss remains in `training/training_log.jsonl`; validation history is
appended directly to `training/validation_log.jsonl`, including explicit skipped
validation markers at checkpoint-only steps. Automated verification: **77 tests
passed, one skipped**, including resume-setting compatibility and the ordered
memory measurements around garbage collection/cache release.

## Authorized completion beyond the runtime budget

On 2026-10-01 the user authorized automatic checkpoint-based resumes after
budget stops until the single epoch completes. `scripts/resume_training_on_budget.py`
ran as a detached watcher under `caffeinate -i`; it left the active
training process unchanged. It waits for `budget_exhausted` and release of the
writer lock, then resumes the recorded checkpoint with the same recovery
configuration. Each invocation retains the 13.5-hour budget. It exits on
completion or failure and refuses repeated resumes without checkpoint progress.
Actual errors are not automatically retried.

Watcher status: `budget_resume_status.json`; launch: `budget_resume_launch.json`;
resume history: `budget_resume_events.jsonl`; console: `budget_resume_console.log`
at the experiment root. Each continuation has a separate
`training_auto_resume_from_NNNNNN.log`. Training/validation/memory logs continue
in the existing training directory, with unchanged memory allocation.

## Completed-run diagnostics

[Open the training diagnostics report](../results/finetuning/gbdna300m_exon_infonce_v002/training/diagnostics/README.md)
for learning curves, validation accuracy, pre-clipping gradient norms, memory
cleanup, runtime versus sequence length, configured learning rate and overlaid
failed attempts. Each figure is saved as PNG and PDF alongside CSV metric
exports, summary JSON and source-log checksums. These artifacts are local under
`training/diagnostics/` and may be absent from a public clone where results are ignored.

Regenerate with `python -m scripts.report_training_diagnostics` followed by the
experiment directory, using the project virtual environment. This is CPU-only
and does not load model weights or change training artifacts.
