# GC/length-matched Smith–Waterman neighborhood null

The aim is to ask whether embedding neighbors share more nucleotide sequence
similarity than expected after controlling approximately for GC composition and
CDS length. The existing cosine graph and UMAP are held fixed. This is a
conditional random-label comparison, not a read-mapping accuracy measurement or
proof of evolutionary homology.

## Completed results

All **100 permutations** and final plots are complete. The accumulated permutation
runtime was **3.75 hours**, excluding final figure export. The
[retained results snapshot](sw_stratified_null_results.json) contains exact numbers
and input/scoring fingerprints; large result artifacts remain in the ignored
`results/` directory.

A gene's score is the unweighted **mean across its graph neighbors**. “Global
median” means the median of those 19,217 gene-level means, not a median over
all graph edges and not a median within each gene's neighborhood.

| Global statistic | Observed | Null center | Null central 95% interval | Observed / null | Enrichment p |
|---|---:|---:|---:|---:|---:|
| Mean of gene scores | 0.026254 | 0.012179 | 0.012116–0.012239 | 2.16× | 0.0099 |
| Median of gene scores | 0.013341 | 0.010849 | 0.010820–0.010876 | 1.23× | 0.0099 |

For the mean test, the null center averages the 100 permutation means. For the
median test, compute a median across genes separately for each permutation,
then summarize those 100 medians: their median is **0.01084912**, their mean is
0.01084893, and their range is 0.01081450–0.01088399. No permutation reached the
observed global median of **0.01334088**. Both tests use the same plus-one upper-tail
p-value, `(1 + exceedances) / 101`; both have zero exceedances. The median
comparison was added after the initial mean analysis using saved null arrays,
without new SW alignments. It is an exploratory robustness summary.

The observed median is **23% higher** than its matched background. Together with
the 2.16-fold mean enrichment, this supports preservation of local DNA sequence
similarity beyond this conditional null. The stronger mean effect is consistent
with a high-similarity tail; it should not be substituted for the median effect.
Neither result quantifies percent identity or proves evolutionary homology.

| Cluster | Genes | Observed mean | Null mean | Fold enrichment |
|---|---:|---:|---:|---:|
| 0 | 8,244 | 0.024573 | 0.012767 | 1.92× |
| 1 | 8,081 | 0.019540 | 0.010984 | 1.78× |
| 2 | 1,946 | 0.026493 | 0.014148 | 1.87× |
| 3 | 643 | 0.130139 | 0.015383 | 8.46× |
| 4 | 303 | 0.029085 | 0.008596 | 3.38× |

All five cluster mean tests have empirical p = 0.0099; these are descriptive,
post-selection comparisons. Cluster 3 has the largest enrichment, consistent
with the earlier zinc-finger-family characterization. Cluster 4 also shows mean
enrichment despite its low raw median; this does not establish a cluster-specific
median effect. Across genes, 14,474/19,217 (75.3%) have positive excess over their
own null mean. This is an effect-direction summary, not a significance count.

The null controls approximate GC/length composition and exact degree, but does
not remove repeat similarity or guarantee equivalent continuous covariates.
The raw negative cosine/SW association within selected graph edges still rules
out interpreting cosine as a uniformly calibrated proxy for this SW score.
The UMAP is a display, not the space in which neighbors were selected. These
results motivate subsequent read-retrieval evaluation; they do not establish
zero-shot mapping accuracy.

## Prespecified design

- Reference: the same 19,217 selected CDS sequences and 461,969 undirected graph
  edges used for the observed SW analysis.
- Permutations: 100, seed 20260929. The first two are a timed pilot and count
  toward the total. Timing does not select or discard permutation outcomes.
- Strata: five marginal GC quantile bins × five marginal log-CDS-length quantile
  bins × **exact graph degree**. Equal covariate values stay in the same bin.
  No shuffling across strata, and no stratification by cluster or observed SW.
- Each shuffle is a bijection assigning genes/sequences to graph positions.
  Every gene is followed to its assigned position; its null statistic is the
  unweighted mean SW similarity to all neighbors there. Exact degree matching
  preserves the number of neighbors for that gene. This avoids comparing a
  gene's observed statistic with a different focal gene's null distribution.
- All edge scores are recomputed using actual shuffled sequences. The scoring,
  strand search and normalization are identical to the observed analysis:
  match +2, mismatch −3, N comparisons 0, affine gap cost `5 + (k−1)`, maximum
  over forward and reverse-complement orientation, and similarity
  `2 × SW / (self_score_A + self_score_B)`. Distance is `1 − similarity`.
- Native Parasail CPU SIMD, eight threads; this alignment operation does not
  use MPS. Reverse complements are precomputed once without changing scoring.

The selected strata give 17,515 genes at least five possible positions; 439
have singleton strata. A singleton focal gene cannot move, but its neighbors
can still change. Restricted strata are flagged and should be interpreted
cautiously. Quantile matching is approximate: especially within the longest
length bin, pairs can still differ substantially in length. The output records
bin boundaries, stratum sizes, and actual mean covariate shifts per shuffle.
Exact degree matching deliberately makes this a more conditional null than
GC/length matching alone. It can retain some genuine family structure, making
excess similarity harder to detect.

## Run or resume

From the project root, with the existing environment and native Parasail library:

```bash
.venv/bin/python -m scripts.evaluate_alignment_permutations \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30 \
  data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa \
  results/gb_dna_300m_cds_2026-09-28/sw_neighborhood \
  results/gb_dna_300m_cds_2026-09-28/sw_stratified_null \
  --permutations 100 --pilot 2 --bins 5 --seed 20260929 --workers 8
```

Use `--stop-after 2` to stop after the pilot, then rerun the command without
that option to continue. Each completed permutation has an atomic checkpoint
and checksum. An interrupted permutation is recomputed; completed permutations
are reused. Resume rejects changed input/scoring/settings and corrupt checkpoints.
A writer lock prevents two processes writing the same analysis simultaneously.

## Outputs and interpretation

Everything is written under
`results/gb_dna_300m_cds_2026-09-28/sw_stratified_null/`:

| File | Meaning |
| --- | --- |
| `manifest.json` | Current status, input fingerprints, settings, completed permutations and timings |
| `pilot.json` | Measured timing and projected total/remaining runtime after two permutations |
| `strata.csv`, `strata_qc.json` | Gene positions available for shuffling; covariate bin boundaries and mixing QC |
| `permutation_000.npy` … `permutation_099.npy` | Null neighborhood means, each in original graph/database gene order |
| `gene_sw_null.csv` | Observed score/distance, null mean/SD/central 95% interval, excess similarity, exceedance count, empirical p and exploratory BH q |
| `cluster_sw_null.csv`, `summary.json`, `report.md` | Global and original-cluster summaries and descriptive enrichment scores |
| `sw_null_umap.png/pdf` | Observed similarity, excess similarity, and −log10 enrichment p on the same UMAP |
| `global_median_summary.json`, `global_median_permutations.csv` | Post hoc global median comparison and all 100 permutation medians |
| `sw_null_qc.png/pdf` | Global null distribution and gene-level observed-versus-null comparison |

Summary tables and plots are generated only after all 100 permutations finish.
Check `manifest.json` for `status: complete`; the existence of the directory or
pilot output does not mean the full analysis has finished. Only per-gene means
are retained for each shuffle, avoiding a 20k × 20k matrix or a large store of
46 million individual edge scores.

For enrichment, the one-sided Monte Carlo p-value is
`(1 + number of null statistics >= observed statistic) / 101`. The same statistic
(mean or median) is used on both sides of each comparison. Ties count as exceedances
(with a conservative 1e-12 floating-point tolerance). Testing unusually small
SW distance gives the same tail. The plus-one convention avoids zero p-values;
see [Phipson & Smyth (2010)](https://pubmed.ncbi.nlm.nih.gov/21044043/).

The smallest attainable p is 1/101 ≈ 0.0099. With 19,217 genes and dependent
neighborhoods, 100 permutations are an exploratory screen, not precise
individual-gene significance estimates. BH q-values are supplied as exploratory
summaries, not a guarantee of FDR control under this graph dependence. Focus
on excess similarity and null intervals; those intervals describe null variation,
not confidence intervals for a biological effect. Cluster summaries are also
descriptive because clusters were selected from the same embedding graph.

A positive excess supports preservation of local sequence similarity beyond
this particular matched background. It does not establish sensitivity to all
homologous genes, nor zero-shot WES read-retrieval performance. Those require
separate evaluation.

## Verification

The implementation is covered by tests for strata/degree preservation,
reproducible shuffles, tracking gene identity through graph positions,
zero-edge inclusion, empirical p-value ties, BH arithmetic, exact scoring
agreement, resumable checkpoints, and provenance/checksum rejection. The full
suite passed with 62 tests and one existing skip. Plot export was smoke-tested
and visually checked with synthetic data; those are not study results.

For the first real pilot permutation, neighborhood means for GAGE12B, TMA7B,
and PSMG4 were independently recomputed through the original SW comparison
API. The maximum difference from the saved null means was 3.47e-18, and exact
degree preservation was verified. Details are saved in `pilot_verification.json`.

## Timed pilot (2026-09-29)

Two full graph shuffles completed in 135.43 and 135.56 seconds, averaging
135.50 seconds per permutation with eight CPU threads. This projects to
3.76 hours for all 100, or 3.69 hours for the remaining 98, excluding final plot
export and allowing for changes in machine load. The full run was then resumed
in the background with idle sleep prevention and subsequently completed all 100
permutations. The completed results are summarized above; these pilot timings
are retained as the original runtime estimate.

The first shuffle retained 9.72% of genes at their original positions (including
singletons and random fixed points). Mean absolute GC shift was 0.0210 and mean
absolute log-length shift was 0.2087. These diagnostics quantify the approximate
covariate matching; the null does not exactly preserve continuous GC/length.

```bash
.venv/bin/python -c 'import json; m=json.load(open("results/gb_dna_300m_cds_2026-09-28/sw_stratified_null/manifest.json")); print(m["status"], len(m["completed"]), "/", m["signature"]["permutations"])'
tail -n 5 results/gb_dna_300m_cds_2026-09-28/sw_stratified_null/run.log
```
