# NA19239 zero-shot 100-bp read retrieval

## Completed experiment

The first experiment used **935 reads** from 12 targets. Each read was embedded
in both orientations using the exact GB.DNA-300M configuration of the reference
vector database. No weights, labels, or reference vectors were trained or tuned
on these reads. Retrieval used cosine similarity against one pooled CDS vector
per gene.

| Callable labels | Correct / reads | Top-1 accuracy | Macro sensitivity | Top-5 accuracy |
|---|---:|---:|---:|---:|
| All 19,217 HGNC genes | 0 / 935 | **0.00%** | 0.00% | 0.75% |
| Only the 12 target genes | 307 / 935 | **32.83%** | 31.20% | 86.52% |

Full per-gene sensitivity, specificity, precision, F1, and counts are retained
in the [all-HGNC report](na19239_retrieval_all_hgnc.md) and
[12-target report](na19239_retrieval_targets_only.md). The
[machine-readable result snapshot](na19239_read_retrieval_results.json) contains
metrics, quality checks, selection counts, and provenance without requiring the
large ignored result files.

## Read selection and truth labels

Input is the verified NA19239 WES CRAM, decoded directly with pysam and the exact
IGSR GRCh38DH reference. There is no intermediate BAM. Selected transcript CDS
blocks come from the same Ensembl/HGNC extraction that produced the embedded
reference. Chromosome aliases (`1` / `chr1`) are resolved against the CRAM header.

Eligibility is primary mapped alignment, nonduplicate, vendor QC pass,
**MAPQ ≥30 and not 255**, exactly **100M** and 100 bases, A/C/G/T sequence,
qualities present, and an AS tag. The entire alignment must lie inside one CDS
block of the selected transcript. Reads overlapping another gene's selected CDS
are excluded. The gene's eligible AS maximum is then determined; a uniform
reservoir sample of up to 100 reads is drawn **only from that maximum-AS tier**.
Lower-AS reads are not used to fill shortfalls. The fixed sampling seed is
20260930; read identity includes read group, query name and mate.

| Gene | Eligible reads | Highest AS | Reads at highest AS | Sampled |
|---|---:|---:|---:|---:|
| TP53 | 158 | 100 | 136 | 100 |
| KRAS | 79 | 100 | 73 | 73 |
| HLA-A | 103 | 100 | 29 | 29 |
| HLA-B | 192 | 100 | 128 | 100 |
| HLA-C | 56 | 100 | 4 | 4 |
| EGFR | 471 | 100 | 392 | 100 |
| MYC | 470 | 100 | 390 | 100 |
| BRCA1 | 2569 | 100 | 2309 | 100 |
| BRCA2 | 3026 | 100 | 2695 | 100 |
| CDKN2A | 37 | 100 | 29 | 29 |
| BRAF | 321 | 100 | 294 | 100 |
| MDM2 | 260 | 100 | 250 | 100 |

Truth is **alignment-derived**, not independent biological validation. It applies
to the selected CDS annotation, not every transcript or all potentially homologous
loci. Mates are separate read observations. Duplicate flags are excluded, but
identical sequences and both mates of a template can remain: 935 unique read IDs,
866 unique sequences, and 894 distinct templates were sampled. Reads are thus
not guaranteed statistically independent.

“M30” is operationalized as mapping quality ≥30. No minimum **base**-quality
threshold was imposed in the primary sample. The observed minimum MAPQ was 38;
45 reads had mean base quality below Q30, including three below Q20 (minimum
mean Q6). A post-selection **mean-base-Q≥30 sensitivity subset** retained 890 reads:
all-HGNC accuracy remained **0/890**, and target-only accuracy was **294/890
(33.03%)**. This is a filtered subset, not a replacement sample of 100 per gene.

All 935 selected reads had **AS=100 and NM=0**. An independent check mapped each
read's genomic coordinates into its selected CDS, handled both gene strands,
and confirmed **935/935 exact 100-base substrings** of the embedded reference
FASTA. Thus this negative result is not explained by the sampled sequences being
absent from the selected CDS. AS/NM perfection was observed, not an added NM filter.

## Embedding and retrieval

- Reference: 19,217 gene vectors, dimension 1,024. Each represents the length-weighted
  mean of raw CDS-context embeddings followed by L2 normalization; context payload
  is at most 3,998 bases.
- Model: `genbio-ai/GB.DNA-300M`, revision
  `a833c5602b0dad0728c0aeeecaf43086ebeb96ca`, float32, SDPA, nucleotide-token mean
  pooling. The read/reference model fingerprints match exactly, including tokenizer
  and package versions.
- Queries: original sequenced orientation and its reverse complement are encoded
  separately, then L2 normalized. **True gene strand is never used** to orient or
  select embeddings. For each candidate gene, the score is the larger cosine of
  the two orientations. The highest candidate score wins; exact ties use database
  row order. Both candidate universes use the same query vectors.
- MPS encoded 1,870 oriented sequences (935 reads × two) in **24.21 seconds**,
  excluding initial model loading, with batch size 32.
- Predictions include nearest gene, cosine, winning orientation, top-1/top-2 margin,
  true-gene rank, and top five candidate symbols. Classification uses original
  1,024-dimensional vectors, never UMAP coordinates.

Per-class sensitivity is `TP/(TP+FN)`, specificity is `TN/(TN+FP)`, precision is
`TP/(TP+FP)`, and F1 is `2TP/(2TP+FP+FN)`. Undefined denominators are reported as
missing. Specificity is one-versus-rest **among these sampled target genes**;
it does not measure rejection of arbitrary genomic backgrounds. Near-100%
specificity in the all-HGNC report mostly reflects the model almost never calling
the target genes, rather than useful mapping performance. Macro sensitivity gives
each represented target gene equal weight; overall accuracy weights reads equally.

## A single fixed reference UMAP

The earlier run saved coordinates but not a reusable fitted model. We recovered
the reference-only fit with the original parameters (cosine, 30 neighbors,
min_dist 0.25, spectral initialization, seed 42, one fitting thread) and saved it
as `results/gb_dna_300m_cds_2026-09-28/plots/reference_umap.joblib` with a
`reference_umap_manifest.json` containing matrix/order hashes and package versions.
The recovered reference coordinates are **exactly identical** to the original
ones (maximum absolute difference 0).

All 12 existing UMAP figure artifacts (PNG/PDF/SVG variants across gene,
composition, cluster, observed SW and matched-null views) were regenerated from
this fit. Graph, gene-score and null-statistic checksums remained unchanged;
`reference_umap_refresh.json` records the audit. Future clustering requires the
saved fit instead of silently fitting another projection.

Both orientations of every query are passed to the saved model's **`transform`**,
without any fit or refit involving queries. Reference coordinates are checked
unchanged after transformation. The input batch/order and model checksum are
recorded; UMAP's numerical transform depends on its saved parameters, software
versions and query batching, so reproduce this exact input batch for the same
query coordinates. The main overlay shows one point per read, using the
orientation chosen by all-HGNC retrieval, colored by true gene origin. This
orientation choice uses predictions, not truth. Both orientations' coordinates
are retained in CSV/NumPy for inspection. Stars label the 12 reference genes.

The serialized model is a **trusted local artifact**; joblib/pickle files from
untrusted sources should not be loaded. Existing figures remain in the same
reference frame; a changed matrix or incompatible saved environment is rejected.

## Interpretation and checks

The earlier matched-null analysis showed local **gene-to-gene** sequence-similarity
enrichment. This experiment shows that such enrichment does not by itself yield
successful **100-bp-read-to-pooled-CDS** top-1 mapping across all genes. Restricting
the callable labels reveals some discrimination (32.83%), but cannot establish
whole-reference mapping performance.

Independent controls re-embedded full CDS for TP53, KRAS and CDKN2A on CPU using
the same model configuration. All three retrieved their own saved gene vector,
with cosine within about 1.2e-7 of 1. Three independent scalar-dot-product checks
also reproduced the read predictions. These controls and the exact-substring
checks support the integrity of the sequence/embedding/retrieval path.

Whole-CDS pooling versus 100-bp queries is a plausible source of representation
mismatch, not a demonstrated sole cause. Local, length-matched reference windows
would be a separate experiment. HLA-C has only four reads, making its per-class
rate especially unstable. This study neither estimates genome-wide specificity
nor resolves pretraining overlap. No confidence claims based on independent
reads are made.

Code verification: **68 tests passed, one existing test skipped**. Coverage includes
highest-AS CRAM sampling, overlap ambiguity, confusion-matrix denominators,
strand-aware retrieval, and persistence/immutability of the reference UMAP.
The query overlay and confusion plots were visually checked.

## Outputs

All local study artifacts are under `results/na19239_gb_dna_300m_retrieval/`:

```text
selection/                 # reads.jsonl, reads.fa, counts CSV, AS/filter audit and source hashes
embeddings/                # [2 orientations, 935 reads, 1024 features] vectors and model manifest
evaluation/
  all_hgnc/                # metrics, per-gene CSV, predictions, confusion CSV, report, Q30 subset
  targets_only/            # Same deliverables with 12 callable labels
  query_reference_umap.*   # Fixed-reference overlay: PNG, PDF, SVG
  query_umap_coordinates.* # Both orientations: CSV and NumPy
  confusion_matrices.*    # Full and restricted confusion plots (PNG/PDF)
  manifest.json            # Query/reference hashes and fixed-transform provenance
verification.json          # Exact CDS matches, full-CDS controls, quality and identity checks
```

The all-HGNC confusion **plot** groups non-target predictions into “Other HGNC”
for readability; `confusion_long.csv` preserves every actually called symbol.

## Reproduce from the project root

```bash
# Fit once (new study), or redraw from an existing saved reference fit:
MPLCONFIGDIR=results/.cache/matplotlib NUMBA_CACHE_DIR=results/.cache/numba \
.venv/bin/python -m llm_mapper.gene_umap \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/plots --replace

# For this recovered identical fit, refresh and audit every earlier UMAP view:
MPLCONFIGDIR=results/.cache/matplotlib \
.venv/bin/python -m scripts.refresh_reference_umap_views \
  results/gb_dna_300m_cds_2026-09-28

.venv/bin/python -m scripts.evaluate_read_retrieval select \
  data/wes_100bp_1000genomes/NA19239/source/NA19239.alt_bwamem_GRCh38DH.20150826.YRI.exome.cram \
  data/reference/GRCh38-IGSR-decoding/GRCh38_full_analysis_set_plus_decoy_hla.fa \
  data/reference/GRCh38-ensembl-110/hgnc_cds/cds_blocks.jsonl \
  results/na19239_gb_dna_300m_retrieval/selection --per-gene 100 --mapq 30 --seed 20260930

HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.evaluate_read_retrieval embed \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/na19239_gb_dna_300m_retrieval/selection \
  results/na19239_gb_dna_300m_retrieval/embeddings --device mps --batch-size 32

MPLCONFIGDIR=results/.cache/matplotlib NUMBA_CACHE_DIR=results/.cache/numba \
.venv/bin/python -m scripts.evaluate_read_retrieval evaluate \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/na19239_gb_dna_300m_retrieval/selection \
  results/na19239_gb_dna_300m_retrieval/embeddings \
  results/gb_dna_300m_cds_2026-09-28/plots \
  results/na19239_gb_dna_300m_retrieval/evaluation

HF_HOME=data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.verify_read_retrieval \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/na19239_gb_dna_300m_retrieval \
  data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  data/reference/GRCh38-ensembl-110/hgnc_cds/cds_blocks.jsonl --device cpu
```

Selection and embedding reuse matching completed artifacts and reject mismatched
provenance. To change inclusion rules, use a new study output directory. Evaluation
regenerates plots from the saved model and cached query embeddings.
