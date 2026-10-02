# Gene embedding Louvain analysis

Completed with `llm_mapper/gene_clusters.py`; see the [methods and rerun command](gene_clustering.md).
Local deliverables: [cluster-colored UMAP](../results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/cluster_umap.png),
[QC figure](../results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/cluster_qc.png), and
[top-20 plots for all clusters](../results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/top_genes_all_clusters.pdf).
These large generated artifacts live under the Git-ignored `results/` directory.

19,217 genes; 5 clusters; resolution 0.3; cosine kNN k=30.

GC eta-squared: 0.559; log10 CDS length eta-squared: 0.023.
Second-seed adjusted Rand index: 0.671; sampled cosine silhouette: 0.15952979028224945.

Cluster IDs are ordered by size. Centrality ranks graph hubs within the cluster, not expression or biological importance.
GC/length eta-squared is the fraction of feature variance explained by cluster membership, not a causal estimate.
UMAP coordinates are used only for display. Axis orientation is arbitrary. The graph uses original embeddings.
Composition and high background cosine similarity can drive clusters; representatives are hypotheses for biological review, not enrichment results.
Resolution and neighbor count affect communities; a second seed is a limited stability check, not a parameter robustness study.

| Cluster | Genes | Median GC | Median CDS bp | Top 5 central genes |
|---|---:|---:|---:|---|
| 0 | 8244 | 0.602 | 1263 | RYR1, DUOX2, TET3, ARHGEF17, PREX1 |
| 1 | 8081 | 0.462 | 1395 | INPP5B, RAPGEF2, PPP3CA, PARP4, RYR2 |
| 2 | 1946 | 0.521 | 972 | SLC38A5, SLC36A2, OR10D3, SLC5A2, OR10V1 |
| 3 | 643 | 0.457 | 1665 | ZNF235, ZNF439, ZNF615, ZNF12, ZNF311 |
| 4 | 303 | 0.439 | 2076 | KIF15, CDK5RAP2, CNTLN, CEP152, CCDC144A |
