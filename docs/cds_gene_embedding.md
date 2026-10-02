# Canonical CDS reference, gene vector database and UMAP

This stage builds one DNA embedding per reviewed HGNC protein-coding gene for
visualization and later retrieval experiments. It uses **CDS**, not full genomic
gene spans. Full-span extraction and NA19239 CRAM QC remain separate, preserved
artifacts. No WES reads are mapped by this stage.

## Completed — 2026-09-29

All **19,217 gene vectors and 20,852 context vectors** are complete. The MPS build
finished at 17:56:27 UTC, with 7,808.5 seconds (about 2.17 hours) accumulated in
committed embedding blocks. This excludes pauses, benchmarking and plotting.
The 1,202 checkpoint blocks passed integrity verification, and both matrices are
finite and unit-normalized. All 12 requested highlight genes appear in the UMAP.
PNG/PDF/SVG plots, coordinates, composition diagnostics and original-space
neighbors are available under `results/gb_dna_300m_cds_2026-09-28/plots/`.

The job resumed at gene 12,368 after the matched-length benchmark, retaining batch
size 2; the manifest records execution history. Subsequent clustering is described
in [the Louvain analysis guide](gene_clustering.md).

The current test suite passes **49 tests, with one existing skip**. Coverage
includes transcript selection, negative-strand extraction, complete context
coverage, weighted pooling, checkpoint resume/corruption rejection, and a real
small GB encoder comparison of eager and SDPA attention.

```sh
tail -f results/gb_dna_300m_cds_2026-09-28/run.log
```

## Reference selection and extraction

The reviewed 19,217 HGNC gene loci are matched back to the same Ensembl 110 GTF
and indexed primary-assembly GRCh38 FASTA. Each gene selects one transcript in
this order: MANE Select, Ensembl canonical, then longest annotated CDS. Ties use
CDS length and transcript ID. For this reference, all **19,217 genes** have a CDS:
**18,981 MANE Select** and **236 Ensembl canonical**; no longest-CDS fallback.

The result contains **33,442,183 coding bases**, compared with 1,318,920,029 bases
in the full genomic gene spans. Median CDS length is 1,290 bases; the longest is
107,973 bases. **17,951 genes fit in one 3,998-base context**, and **1,266 need
multiple contexts**: **20,852 contexts total**.

Only the selected transcript's CDS features are used. Shared regions from other
transcripts are not duplicated. Negative-strand CDS blocks are reverse-complemented
individually and concatenated in transcript order. GTF phase is preserved as
metadata; bases are not trimmed from the beginning of every exon. Separate
`stop_codon` features are not appended. The extraction retains the annotation as
provided: 11 selected transcripts carry partial-CDS tags, and 36 CDS lengths are
not divisible by three. These exceptions and transcript tags are retained in the
[extraction audit report](cds_reference_report.json), not silently repaired or excluded.

```text
data/reference/GRCh38-ensembl-110/hgnc_cds/
  genes.fa                 # One strand-oriented, spliced CDS per HGNC symbol
  genes.csv                # Selected transcript, sequence and CDS block mapping
  cds_blocks.bed           # Genomic CDS intervals, 0-based half-open
  cds_blocks.jsonl         # Genomic intervals, CDS offsets, strand and phase
  report.json              # Selection counts, transcript tags, audit and hashes
```

Reproduce in a **new output directory**:

```sh
.venv/bin/python -m llm_mapper.preparation.cds \
  --fasta data/reference/GRCh38-ensembl-110/sources/GRCh38.primary_assembly.fa \
  --gtf data/reference/GRCh38-ensembl-110/sources/GRCh38.ensembl-110.gtf.gz \
  --genes data/reference/GRCh38-ensembl-110/hgnc_genes/genes.csv \
  --output data/reference/GRCh38-ensembl-110/hgnc_cds_reproduction
```

`python -m llm_mapper.extract_gene_cds_sequences` is now a compatibility entry
point for the same modular extractor. It replaces the legacy script's mixing of
transcripts and incorrect negative-strand block ordering, and avoids loading the
whole genome into memory.

## Model context and embedding representation

The pinned GB.DNA-300M checkpoint has `max_position_embeddings=4000`. Its current
tokenizer emits two special tokens, so the safe nucleotide payload is **3,998
bases**, not 4,096. The initial 250-bp proposal was a read-scale representation,
not the model's context limit, and is **not used for this CDS database**.

- Model: `genbio-ai/GB.DNA-300M`, revision
  `a833c5602b0dad0728c0aeeecaf43086ebeb96ca`.
- Inference: MPS, float32, validated GB-specific PyTorch SDPA adapter, batch size 2.
- Within each context: masked mean over nucleotide token hidden states, excluding
  padding and special tokens. Embedding dimension: 1,024.
- Contexts: non-overlapping, up to 3,998 bases, with the final shorter context
  retained. No bases are truncated or sampled away.
- One gene vector: length-weighted mean of **raw, unnormalized** context vectors,
  accumulated in float64, then L2-normalized and stored as float32.
- Context vectors are separately L2-normalized and retained. A single-context
  gene vector equals its normalized context vector.

This is a pooled representation for long CDS sequences, not a model forward pass
with attention spanning the entire long transcript. Gene vectors blend local
signals; their effectiveness for short-read retrieval must be measured.

## MPS verification

[The CDS benchmark](benchmarks/gb_dna_300m_cds_mps.json) compares MPS implementations
with CPU float32 on 100, 150, 250, 1,024 and 3,998-base contexts. SDPA minimum
cosine agreement was **0.99999988**, with maximum relative L2 error **6.51e-7**.
The adapter retains the original learned projections and rotary embeddings and
changes only the attention evaluation to PyTorch SDPA; it does not globally
modify PyTorch or other model backends.

For 24 actual CDS contexts selected across the length distribution, after warmup:

| MPS implementation | Batch | Bases/second | Sampled peak device allocation |
|---|---:|---:|---:|
| Original eager attention | 1 | 3,041 | 9.05 GB |
| SDPA, float32 | 1 | 5,597 | 2.23 GB |
| SDPA, float32 | 2 | 5,651 | 2.23 GB |
| SDPA, float32 | 4 | 5,383 | 3.30 GB |
| SDPA, float32 | 8 | 4,409 | 5.45 GB |

The selected SDPA configuration is about **1.86× faster** than eager MPS on this
sample. That mixed-length test includes padding overhead. A follow-up
[matched-length benchmark](benchmarks/gb_dna_300m_matched_length_mps.json) used
32 actual CDS prefixes at each length and two synchronized repeats. At 3,998 bp,
batches 2/4/8/16 achieved 5,526/5,441/5,499/5,131 bases per second, respectively.
At 1,500 bp, batch 16 improved throughput only about 3.4% over batch 2 while
doubling sampled device allocation. All batch comparisons passed numerical
agreement checks. **Batch 2 remains selected** for the full build.
The earlier [250-bp benchmark](benchmarks/gb_dna_300m_250bp_mps.json)
measured about 3.9× MPS-vs-CPU throughput. Plain float16 failed the chosen 1%
relative-L2 tolerance on that test and was not used. Timing samples are not a
whole-run guarantee; memory figures are sampled allocations, not total system
unified-memory peaks. RSS and device allocations overlap and must not be summed.

## Build, resume and query

The build processes genes in deterministic length order to reduce padding and
commits every 16 genes. A writer lock prevents concurrent writes. Resume verifies
model/source/aggregation fingerprints and the checksums of committed matrix
blocks; interrupted blocks are overwritten. Incomplete databases cannot be
opened for normal retrieval or plotted as a finished reference.

```sh
HF_HOME="$PWD/data/hf-cache" HF_HUB_OFFLINE=1 \
MPLCONFIGDIR="$PWD/results/.matplotlib-cache" \
.venv/bin/python -m llm_mapper.gene_database \
  --fasta data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  --metadata data/reference/GRCh38-ensembl-110/hgnc_cds/genes.csv \
  --output results/gb_dna_300m_cds_2026-09-28/vector_db \
  --device mps --attention sdpa --dtype float32 \
  --chunk-length 3998 --batch-size 2 --genes-per-checkpoint 16 \
  --plot-output results/gb_dna_300m_cds_2026-09-28/plots
```

The same command resumes a stopped build. If plots already exist, omit
`--plot-output` or choose a new plot directory; completed plots are not silently
overwritten. Normal terminal execution can use MPS directly; Codex's sandbox
requires permission for GPU access.

```text
results/gb_dna_300m_cds_2026-09-28/
  benchmark.json
  run.log
  vector_db/
    manifest.json          # Status, model/source hashes, context/pooling policy
    verification.json      # Completed matrix/checkpoint/retrieval integrity audit
    metadata.sqlite        # HGNC/transcript metadata and committed-block checksums
    gene_vectors.npy       # 19,217 × 1,024 float32, normalized
    window_vectors.npy     # 20,852 × 1,024 float32, normalized CDS contexts
    gene_index/            # Compatible exact-cosine index for existing CLI
  plots/
    gene_umap.png / .pdf / .svg
    gene_umap_diagnostics.png / .pdf
    gene_umap_coordinates.csv
    highlight_neighbors_original_space.json
    plot_manifest.json
```

The vector database uses SQLite metadata plus memory-mapped NumPy matrices and
exact cosine search; no external database service or approximate index is needed
for this scale. `GeneVectorDatabase.search_vectors(..., level="gene"|"window")`
returns HGNC symbols and **CDS sequence offsets**, not genomic coordinates.
`search(...)` encodes query sequences with a fingerprint-matched embedder.
The compatibility `gene_index` lets the existing `map` CLI query gene centroids.

The storage and plotting code use the common `SequenceEmbedder` contract.
For another model, choose its backend, checkpoint, supported context length,
tokenization and pooling settings and a new results directory. The explicit SDPA
adapter is currently GB-only; other backends use their own supported attention.
Fingerprints include the attention implementation and prevent mixing incompatible
reference/read configurations. Existing earlier indexes may need rebuilding.

The builder audits the finished matrices before plotting. Re-run this audit
without loading the model or using the GPU:

```sh
.venv/bin/python -m llm_mapper.verify_gene_database \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  --output results/gb_dna_300m_cds_2026-09-28/vector_db/verification.json
```

It checks dimensions, finite unit-normalized vectors, all committed block hashes,
SQLite integrity, complete gene/context coverage and exact-cosine self queries.
These checks validate storage and retrieval, not zero-shot mapping accuracy.

## Visualization

UMAP uses **cosine distance in the original 1,024-dimensional space**, 30
neighbors, `min_dist=0.25`, seed 42 and one CPU job for reproducibility. MPS is
used for transformer inference; this UMAP implementation runs on CPU. No labels
are used to fit the projection. The fitted reference-only model is persisted as
`plots/reference_umap.joblib` with its provenance manifest. Future queries use
its `transform` without participating in fitting. The recovered fit reproduces
the original reference coordinates exactly; all previous UMAP views were
regenerated. See the [read-retrieval study](na19239_read_retrieval.md).

The legacy highlight set is retained: TP53, KRAS, HLA-A, HLA-B, HLA-C, EGFR, MYC,
BRCA1, BRCA2, CDKN2A, BRAF and MDM2. Labels use distinct colors and displacement
with leader lines. The same projection is also colored by CDS length and GC
fraction to expose possible sequence-composition effects. Original-space cosine
neighbors of highlighted genes are exported separately; they are not inferred
from distances on the plot.

```sh
.venv/bin/python -m llm_mapper.plot_umap_genes \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/plots --replace \
  --highlight HLA-A HLA-B HLA-C TP53 BRCA1 BRCA2
```

## Scope of the initial zero-shot WES mapping study

The [initial retrieval experiment](na19239_read_retrieval.md) uses only reads
fully contained in selected CDS blocks and excludes ambiguous CDS overlaps.
CDS excludes UTR exons and alternative-only coding exons; those reads are outside
this experiment. Use the actual capture design if assessing
on-target coverage. Spliced CDS contains exon joins absent from ordinary genomic
WES reads: its context vectors are transcript representations, not contiguous
genomic read windows. Use `cds_blocks.bed`/`cds_blocks.jsonl` for genomic overlap
truth; do not treat a CDS offset as a chromosome coordinate. A later exon-block
reference can avoid artificial joins when testing local-window retrieval.
Compare whole-CDS gene centroids against length-matched exon-block vectors: a
larger model context alone does not establish comparability between a 100-bp
query and a multi-kilobase reference embedding.

Define primary/MAPQ/duplicate rules, multimapping and gene-overlap truth, query
strand handling (GB embeddings are not assumed reverse-complement invariant),
sampling and top-k metrics before evaluating reads. HLA allele/alternative contigs
need a stated policy; the current CDS set uses the reviewed primary gene loci.
Compare against simple sequence baselines and examine gene length/GC confounding.
Zero-shot here means no task-specific fitting on the mapping labels; it does not
establish that the human reference was absent from model pretraining. UMAP
separation alone is not mapping accuracy or evidence of biological function.
