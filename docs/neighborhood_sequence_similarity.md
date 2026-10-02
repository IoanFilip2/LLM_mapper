# Sequence similarity within embedding neighborhoods

This evaluation asks whether neighboring **gene embeddings** also have similar
DNA sequences. It uses the saved cosine kNN graph and overlays one alignment-based
scalar per gene on the existing UMAP. The graph is in original embedding space;
UMAP coordinates are only used for display.

## Completed evaluation — 2026-09-29

All **461,969 unique pairs** were evaluated in both orientations and aggregated
for **19,217 genes**, with no isolates. The accumulated alignment-batch time was
149.8 seconds on eight CPU workers; setup, checkpoint I/O and plotting are extra.
The [retained measured report](neighborhood_sw_results.json) includes integrity
checks and the input/library fingerprints. Tests pass: **57 passed, one existing
skip**. Seven additional full-CDS pairs matched Parasail's non-SIMD reference
implementation exactly, and every exported unweighted/weighted gene mean was
recomputed from the saved sparse matrix.

| Cluster | Genes | Median gene mean SW similarity |
|---|---:|---:|
| 0 | 8,244 | 0.0140 |
| 1 | 8,081 | 0.0113 |
| 2 | 1,946 | 0.0179 |
| 3 | 643 | 0.1130 |
| 4 | 303 | 0.0097 |

Cluster 3 has conspicuously stronger local nucleotide correspondence under this
score, consistent with the earlier zinc-finger-family observations. Cluster 4
does not show similarly high median nucleotide similarity under the whole-CDS
normalization; this does not rule out protein-domain or structural similarities.
The global median gene score is 0.0133, and the maximum is 0.6600. These values
are **not identity percentages**.

An important limitation is visible in the QC: gene scores correlate negatively
with log-CDS length (Spearman −0.726). Among the evaluated graph edges, embedding
cosine and normalized SW have Spearman −0.500. Thus this run does **not** support
a blanket claim that higher embedding cosine means higher normalized DNA
alignment similarity. These associations mix sequence length, gene families,
composition, normalization and selection of graph edges. Gene-score/GC correlation
is +0.146. The subsequent [matched-null evaluation](sw_stratified_null.md) found
global mean enrichment of 2.16× (0.02625 versus 0.01218) and global median
enrichment of 1.23× (0.01334 versus 0.01085), each with empirical p = 0.0099
from 100 permutations. This supports local sequence-similarity enrichment over
that background, while leaving the negative raw cosine/SW association intact.

Local outputs: [SW-colored UMAP](../results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/neighbor_sw_umap.png),
[per-gene CSV](../results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/gene_neighbor_sw.csv),
[QC plots](../results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/neighbor_sw_qc.png), and
[edge cosine/SW comparison](../results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/edge_cosine_sw.png).

## Run or resume

Run from the project root after installing the `alignment`, `visualize`, and
`bam` dependency extras. Apple Silicon installation notes are below.

```sh
MPLCONFIGDIR="$PWD/results/.matplotlib-cache" \
.venv/bin/python -u -m scripts.evaluate_neighbor_alignment \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30 \
  data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  results/gb_dna_300m_cds_2026-09-28/sw_neighborhood \
  --workers 8 --algorithm striped
```

The same command resumes interrupted work. Each SQLite transaction commits a
batch of 256 unique pairs; a writer lock prevents simultaneous runs on the same
output. A manifest fingerprints the reference FASTA, vector matrix, graph, node
order, UMAP coordinates, scoring configuration and native library. Changing any
of these requires a new output directory. `--max-batches` stops after a specified
number of new batches for timing or resume checks. No input sequence is truncated,
sampled or split for alignment. The complete 33.4-million-base CDS collection is
held in memory (the whole human genome is not loaded).

## Alignment definition

The reusable submodule is `llm_mapper/alignment/`; orchestration and sparse
aggregation are in `llm_mapper/neighborhood_alignment.py`. The CLI wrapper is
`scripts/evaluate_neighbor_alignment.py`.

[Parasail](https://github.com/jeffdaily/parasail) implements exact SIMD
Smith–Waterman local alignment with affine gaps. Current parameters:

- Match +2; mismatch −3; gap open 5; gap extension 1.
- A gap of length k costs `5 + (k - 1)`.
- Both `A versus B` and `A versus reverse_complement(B)` are evaluated; take the
  larger score. Forward wins ties. `--forward-only` disables the second alignment.
- Supported alphabet: A/C/G/T/N, case-insensitive. N comparisons score zero and
  do not contribute to the normalization denominator; all-N inputs are rejected.
  The current CDS reference contains only A/C/G/T.
- Use 16-bit SIMD arithmetic when the score bound is safely below overflow,
  otherwise 32-bit; any detected saturation retries in 64-bit arithmetic.
- Score-only execution avoids storing traceback or a full DP table. Computation
  uses native CPU NEON kernels on this Mac, not MPS.

For positive self-scores `self(A) = 2 × number_of_ACGT_bases(A)`, define:

```text
pair_similarity(A,B) = 2 × best_SW(A,B) / (self(A) + self(B))
pair_dissimilarity(A,B) = 1 − pair_similarity(A,B)
```

The similarity lies in [0,1]. Identical unambiguous sequences score 1. This
normalization penalizes a short matching segment within much longer sequences;
it combines local alignment strength with coverage of both sequences. It is
**not percent identity**, an E-value, or a proof of evolutionary homology. Its
complement is not guaranteed to satisfy the triangle inequality.

A secondary score `best_SW / min(self(A), self(B))` is retained to show similarity
relative to the shorter sequence. For example, an exact 1-kb substring of a
10-kb sequence scores 1 by that containment measure but approximately 0.182 by
the primary symmetric normalization. Length and domain architecture therefore
matter to interpretation. The alignments compare our spliced CDS representations,
not unspliced genomic loci or protein sequences.

## Sparse graph computation

The current undirected graph has **461,969 unique edges**. Each pair is computed
once and shared between its two endpoints; both DNA orientations are checked
within that computation. There is no 19,217-by-19,217 alignment matrix.

Let `A` be the saved cosine-weighted adjacency, `B` its binary edge mask, and `S`
the sparse symmetric matrix of alignment similarities at those edges:

```text
primary_gene_score = (S @ ones) / (B @ ones)
weighted_gene_score = ((A.multiply(S)) @ ones) / (A @ ones)
```

The **primary color is the unweighted mean** over all neighbors in the saved
union-symmetrized graph. Degree can exceed 30 because incoming neighbors are
retained. The cosine-weighted mean is exported separately. Zero-alignment-score
edges remain in the degree denominator; otherwise means would be inflated.
Isolated genes receive missing values and gray coloring, not a fabricated score.
Absent sparse entries mean “not evaluated,” not “no sequence similarity.”

Pairs are scheduled by descending sequence-length product so expensive long-CDS
pairs run first. Progress reports include both pair count and theoretical DP-cell
work; neither is a guarantee of remaining wall time. Threaded execution invokes
native code; it does not run Python DP for the full graph.

## Validation and benchmark

An independent affine-gap Python DP implementation tests the SIMD results on
seeded short sequences, mismatches, N bases and gapped examples. Tests also check
reverse complements, length normalization, 16-bit overflow avoidance, sparse
aggregation with zero-score edges/isolates, and checkpoint resume/provenance.

`scripts/benchmark_neighbor_alignment.py` compares striped and scan kernels on
35 actual graph pairs spanning length-product quantiles and the four most
expensive pairs. All raw scores agreed across implementations and 1/4/8 workers.
On this Mac, striped runs took 3.50/1.69/1.68 seconds, versus 5.61/2.66/2.64 seconds
for scan. These are short benchmark timings, not a sustained-throughput guarantee.
The measured report lives at `results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/benchmark.json`.

## Outputs and interpretation

```text
results/gb_dna_300m_cds_2026-09-28/sw_neighborhood/
  manifest.json                 # building/complete status and reproducibility signature
  edge_scores.sqlite            # Raw and normalized pair scores, orientation, batch timings
  neighbor_sw_similarity.npz    # Sparse symmetric scores; node order from graph_nodes.csv
  gene_neighbor_sw.csv          # Every gene's mean SW, distance, weighted/containment means,
                                # degree, cluster, GC, CDS length, and UMAP coordinates
  neighbor_sw_umap.png/.pdf/.svg # Same UMAP colored by primary mean SW similarity
  neighbor_sw_qc.png/.pdf        # Score distribution, cluster boxplots, GC/length diagnostic
  edge_cosine_sw.png/.pdf        # Edge-level cosine versus SW density plot
  cluster_sw_summary.csv        # Cluster sizes, mean/median/p95 gene-neighborhood scores
  output_verification.json      # Recomputed scalar means and reference-alignment spot checks
  qc.json                       # Coverage, score summaries and descriptive correlations
  benchmark.json                # Actual-pair benchmark
  run.log                       # Progress, if stdout was redirected here
```

The UMAP figure includes a fixed [0,1] scale and a clearly labeled contrast view
whose color maximum is the 99th percentile. The top 1% is only color-clipped;
CSV values remain unchanged. QC reports edge cosine/SW and gene-level GC/length
Spearman correlations. Graph edges are dependent observations, so these are
reported descriptively without naive edge-independent p-values.

High values indicate embedding neighbors with strong nucleotide alignment
relative to their lengths. Low values may reflect weak sequence correspondence,
short shared domains, composition-driven embedding similarity, or the strict
whole-CDS normalization. They do not by themselves prove that biological function
is unrelated. These original raw scores use no chance-score subtraction, repeat
masking, or alignment-identity estimation. A subsequent
[100-permutation GC/length/degree-matched null](sw_stratified_null.md) now provides
excess-score and empirical-p comparisons. Its binned matching does not eliminate
all composition or repeat effects, and SW similarity does not prove homology.
This remains distinct from testing WES read mapping.

## Native Apple Silicon installation used here

The automatic Parasail pip build failed during its autotools step. The successful
route used the Python bindings with a separately compiled native library:

```sh
PARASAIL_SKIP_BUILD=1 .venv/bin/python -m pip install parasail==1.3.4 cmake==3.31.10
mkdir -p data/alignment-tools
curl -fL https://github.com/jeffdaily/parasail/archive/refs/tags/v2.6.2.tar.gz \
  -o data/alignment-tools/parasail-v2.6.2.tar.gz
tar -xzf data/alignment-tools/parasail-v2.6.2.tar.gz -C data/alignment-tools
.venv/bin/cmake -S data/alignment-tools/parasail-2.6.2 \
  -B data/alignment-tools/parasail-build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
.venv/bin/cmake --build data/alignment-tools/parasail-build --target parasail -j 8
.venv/bin/python - <<'PY'
from pathlib import Path
import importlib.util
import shutil
package = Path(importlib.util.find_spec('parasail').origin).parent
shutil.copy2('data/alignment-tools/parasail-build/libparasail.dylib', package/'libparasail.dylib')
PY
.venv/bin/python -c 'import parasail; print(parasail.version(), parasail.can_use_neon())'
```

The verified source archive SHA-256 is
`9057041db8e1cde76678f649420b85054650414e5de9ea84ee268756c7ea4b4b`.
The local check returned Parasail C 2.6.2 and NEON available. Native build files
remain under Git-ignored `data/alignment-tools/`; no machine-specific paths or
compiled binaries need to be committed. On platforms with a working Parasail
package installation, the manual build is unnecessary.
