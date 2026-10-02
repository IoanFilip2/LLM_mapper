# GB.DNA-300M synthetic smoke benchmark

These results validate execution, not biological mapping accuracy. Four seeded random 250-bp references and twelve exact-substring reads per length were used, with three timed repeats after warmup. Batch size: 4; float32; masked mean pooling; CPU-built reference index shared across devices. Model loading is excluded.

| Read length | CPU reads/s (median) | MPS reads/s (median) | Speedup | CPU / MPS accuracy |
|---|---:|---:|---:|---:|
| 100 bp | 17.41 | 48.84 | 2.81x | 58.3% / 58.3% |
| 150 bp | 12.23 | 37.66 | 3.08x | 66.7% / 66.7% |
| 250 bp | 7.38 | 24.45 | 3.31x | 100.0% / 100.0% |

The low short-read accuracy even on this artificial task shows why runtime success must not be interpreted as evidence of reliable gene mapping. No held-out biological dataset was supplied.

Memory maxima across all measured repeats:

- CPU: sampled peak process RSS 1.57 GiB; sampled device allocation 0.00 GiB.
- MPS: sampled peak process RSS 0.85 GiB; sampled device allocation 2.07 GiB.

MPS driver allocation and RSS overlap in unified memory and must not be summed. Sampling interval: 10 ms; transient peaks may be missed. These are not total system unified-memory measurements.

Platform: `macOS-26.5.1-arm64-arm-64bit`; system unified memory: 48 GiB.

Model revision: `a833c5602b0dad0728c0aeeecaf43086ebeb96ca`. Exact package versions and all individual repeats are in [CPU results](gb-dna-300m-cpu.json) and [MPS results](gb-dna-300m-mps.json).

Reproduce from the repository root (use new output filenames):

```sh
HF_HOME="$PWD/data/hf-cache" python -m scripts.benchmark_synthetic --device cpu --work-dir data/synthetic-smoke --output cpu.json
HF_HOME="$PWD/data/hf-cache" python -m scripts.benchmark_synthetic --device mps --work-dir data/synthetic-smoke --output mps.json
```

## Canonical CDS context benchmark

The [CDS workflow](../docs/cds_gene_embedding.md) documents the later real-CDS
benchmark at up to 3,998 bases. [Measured results](../docs/benchmarks/gb_dna_300m_cds_mps.json)
validate the opt-in GB SDPA adapter against CPU float32 and select MPS float32,
batch size 2. SDPA achieved about 1.86× the original MPS attention throughput
while reducing sampled device allocation. These measurements are separate from
the synthetic mapping smoke benchmark above.

## Contrastive fine-tuning feasibility

The [v001 MPS training pilot](../docs/contrastive_finetuning.md) includes actual
forward/backward/AdamW work, rather than extrapolating from inference speed.
Twelve updates were discarded after measurement; **full training has not started**.
Peak MPS allocation was 21.94 GB, and the coarse estimate is 108 compute-hours
per epoch for 200,000 triplets (200,000 positive plus 200,000 negative pairs).
See the [retained measurements](../docs/benchmarks/contrastive_gbdna300m_v001.json)
for stratification, the longest-gene stress case, and uncertainty/overhead limits.

The current [v002 exon-contained pilot](../docs/contrastive_finetuning_v002.md)
reduces training to 25,000 triplets across 250 genes. Its central estimate is
11.45 compute-hours, with 21.99 GB peak MPS allocation; full training remains
unstarted. The accepted planning window is 12–14 hours, with a resumable stop
budget at 13.5 hours if the epoch has not finished.
