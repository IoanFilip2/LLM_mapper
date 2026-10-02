# Completed fine-tuned v002 reference and WES evaluation

The complete pipeline finished on **2026-10-02 at 00:09 UTC** (October 1,
17:09 Pacific), using final fine-tuned checkpoint 001563. This is a task-adapted
model evaluated on held-out target genes, not a zero-shot model. The original
base-model results remain unchanged.

## Reference database and visualization

All **19,217 selected CDS genes** and **20,852 contexts** were embedded into
1,024-dimensional normalized float32 vectors. All 1,202 storage checkpoint
blocks passed verification, SQLite integrity was valid, and 12 stored-vector
self-queries passed. The CDS source, tokenizer/pooling recipe, target gene colors,
and cosine-retrieval method match the base experiment; encoder weights differ.

A new reference-only UMAP was fitted and its transform saved. The 935 reads were
projected through that transform without refitting. Separate models produce
separate UMAP spaces: axis positions should not be compared directly between
base and fine-tuned figures. SW scoring and matched-null analysis were omitted.

## Held-out WES retrieval

The exact same **935 reads** were used, with no reselection. All 12 target genes
were excluded from fine-tuning anchors, negative donors and validation.

| Model | Universe | Top-1 | Top-5 | Top-50 | Top-100 | Top-250 |
|---|---|---:|---:|---:|---:|---:|
| Base | all_hgnc | 0.00% | 0.75% | 5.45% | 7.81% | 13.05% |
| Base | targets_only | 32.83% | 86.52% | 100.00% | 100.00% | 100.00% |
| Fine-tuned v002 | all_hgnc | 0.53% | 3.10% | 14.33% | 21.50% | 29.20% |
| Fine-tuned v002 | targets_only | 38.40% | 86.63% | 100.00% | 100.00% | 100.00% |

All-gene top-1 improved from **0/935 to 5/935**; target-only top-1 improved from
**307/935 to 359/935**. Top-250 all-gene retrieval improved from **122/935 to
273/935**. These are descriptive paired-cohort comparisons, not significance
tests; identical sequences and shared templates mean reads are not independent.
Top-50/100/250 in the 12-label universe include every candidate and are trivially
100%. Specificity is conditional on other sampled target genes, not all genomic
origins.

Fine-tuning improves the rank of the correct gene, but **0.53% all-gene top-1
accuracy remains inadequate for reliable mapping**. The result does not establish
that homology, rather than composition or other sequence features, drives the
improvement. The representation still pools whole CDS while queries are 100 bp.

Verification confirmed all 935 reads exactly match selected CDS substrings
(AS=100, NM=0). Independent CPU full-CDS controls for TP53, KRAS and CDKN2A
retrieved themselves with cosine approximately 1. Three independently recomputed
query dot products agreed with saved retrieval. Q30 subset results and per-gene
sensitivity, specificity, precision/F1 are retained in the evaluation folders and
compact documentation snapshot.

## Clusters and representative genes

Exact cosine kNN (k=30), union symmetrization and Louvain resolution 0.3 produced
**4 clusters**, 427,360 edges, one connected component and no isolated genes.
Cluster IDs are assigned anew by size; they do not identify equivalent clusters
in the base-model analysis.

| Cluster | Genes | Median GC | Median CDS bp | First five central genes |
|---|---:|---:|---:|---|
| 0 | 9,652 | 0.589 | 1269 | RYR1, MYO10, ITGAE, ITGA3, MYO7A |
| 1 | 6,435 | 0.444 | 1479 | LYST, KMT2C, REV3L, MYCBP2, SETD2 |
| 2 | 2,395 | 0.526 | 972 | SLC5A11, SLC38A5, OR10J5, OR4D5, OR11G2 |
| 3 | 735 | 0.485 | 1620 | ZNF44, ZNF12, ZNF836, ZNF81, ZNF699 |

GC eta-squared is **0.510**, versus **0.024** for log10 CDS length, indicating
substantial composition separation. Sampled cosine silhouette is **0.200**.
The second seed yielded five clusters with adjusted Rand index **0.801**;
therefore the four-cluster partition is not invariant to seed.

Cluster 2 representatives include transporters and olfactory receptors; cluster
3 representatives include many zinc-finger genes. These are descriptive patterns,
not functional conclusions based on representative names alone. Full-cluster
HGNC-family Fisher tests, BH correction, MANE/length analyses and top-50 annotated
gene tables were generated for all four clusters. Centrality ranks graph hubs,
not expression or biological importance.

## Where to view outputs

- [Reference plots](../results/gb_dna_300m_exon_infonce_v002_cds_2026-10-01/plots/)
- [Cluster report](../results/gb_dna_300m_exon_infonce_v002_cds_2026-10-01/louvain_r0.3_k30/report.md)
- [Cluster gene analyses](../results/gb_dna_300m_exon_infonce_v002_cds_2026-10-01/louvain_r0.3_k30/gene_analysis/)
- [Retrieval comparison](../results/na19239_gb_dna_300m_exon_infonce_v002_retrieval/evaluation/comparison.md)
- [Read overlay on reference UMAP](../results/na19239_gb_dna_300m_exon_infonce_v002_retrieval/evaluation/query_reference_umap.png)
- [All-gene report](../results/na19239_gb_dna_300m_exon_infonce_v002_retrieval/evaluation/all_hgnc/report.md)
- [12-target report](../results/na19239_gb_dna_300m_exon_infonce_v002_retrieval/evaluation/targets_only/report.md)
- [Retained results snapshot](benchmarks/finetuned_v002_reanalysis_results.json)

Large results folders are local and ignored by Git; the compact snapshot remains
available with repository documentation. See the [workflow guide](finetuned_v002_reanalysis.md)
for commands and the [completed training report](contrastive_finetuning_v002_results.md)
for model provenance and diagnostics.
