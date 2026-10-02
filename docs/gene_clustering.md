# Louvain analysis of gene embeddings

This exploratory stage takes a **completed gene vector database**, clusters its
original embeddings, and exports representative genes for biological review. It
uses no model inference or mapping labels. CPU graph analysis is appropriate at
this scale; MPS is not needed again.

The completed run found **five clusters**. See the [measured cluster report](gene_clustering_results.md)
and [QC metrics](gene_clustering_qc.json). Cluster membership explains 55.9% of GC
variance and 2.3% of log-CDS-length variance; sampled silhouette is 0.160, and
second-seed agreement is ARI 0.671. This supports the visual GC gradient while
showing that membership boundaries are not fully stable across seeds.

The [cluster 3/4 follow-up](cluster_3_4_biology.md) tests MANE selection and long-CDS
enrichment, characterizes HGNC gene families, and lists all 50 central genes from
each cluster with their names, lengths and transcript-selection status.

The [Smith–Waterman neighborhood evaluation](neighborhood_sequence_similarity.md)
uses this saved graph to quantify nucleotide similarity along its edges and
color the same UMAP by each gene's mean neighbor alignment score.

## Run from the project root

```sh
.venv/bin/python -m pip install -e '.[visualize]'
MPLCONFIGDIR="$PWD/results/.matplotlib-cache" \
.venv/bin/python -m llm_mapper.gene_clusters \
  results/gb_dna_300m_cds_2026-09-28/vector_db \
  results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30 \
  --resolution 0.3 --neighbors 30 --seed 42 --top-n 20 \
  --coordinates results/gb_dna_300m_cds_2026-09-28/plots/gene_umap_coordinates.csv
```

Choose a new output directory for another analysis run. The projection must be
the saved reference-only UMAP fit; clustering does not fit another UMAP. If
`--coordinates` is omitted, the database sibling `plots/` directory is used. Its
`plot_manifest.json`, fitted-model checksum, gene order and coordinates must match
the database. Generate the reference fit with `llm_mapper.gene_umap` first. The same projection
is reused across cluster, GC and CDS-length panels. Arbitrary UMAP axis rotations
mean that “top” and “bottom” have no inherent biological interpretation.

## Graph and clustering

The graph uses **exact cosine nearest neighbors in the original 1,024-dimensional
space**, not proximity on UMAP. Each gene selects its 30 nearest other genes;
exact ties use database row order. Computation is blocked (`--block-size 256`) to
avoid allocating a full all-pairs matrix. The undirected graph retains an edge
if either endpoint selected the other, with the maximum directed cosine weight.
Nonpositive similarities and self loops are excluded. All genes, including
isolates, remain represented. The saved sparse graph can be inspected or reused
in subsequent studies.

[NetworkX Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
optimizes weighted modularity at resolution 0.3 and seed 42. Low resolution favors
larger communities; this value is a study setting, not a universal biological
scale. Cluster IDs start at zero, ordered by descending size, with ties resolved
by the smallest database row index. Resolution, neighbor count, and seed are
recorded along with package versions and the database fingerprint.

## Representative genes

Within each cluster C, cosine degree centrality is:

```text
centrality(i) = sum(cosine(i,j) for graph neighbors j in C) / (size(C) - 1)
```

Singletons receive zero. Rankings are by summed internal cosine weight (equivalent
to the normalized score within a cluster), then mean cosine to **all** other
cluster members, then HGNC symbol. Thus the top genes are well-connected hubs in
the cluster subnetwork. This does not measure expression, biological importance,
or statistical enrichment. When edge cosines are very similar, this ranking is
largely driven by the number of internal neighbors. The dense mean-to-peers
cosine is exported separately, making that distinction inspectable.

Each cluster gets a horizontal top-20 plot; clusters smaller than 20 retain every
gene. The same figures are collected in a multipage PDF. CSV exports retain all
genes and their ranks, degrees, centrality, mean peer cosine, GC, CDS length,
HGNC IDs and selected transcript IDs.

## QC and interpretation

- Gene/vector integrity is checked before graph construction.
- Global graph components, isolates, singleton communities and per-community
  connectivity expose fragmented partitions.
- Weighted modularity uses the requested resolution. Mean within-cluster cosine
  is computed over all distinct pairs without materializing their full matrix.
- A cosine silhouette score uses a fixed random sample of up to 2,000 genes.
  Undefined cases are stored as null, not as zero.
- A second Louvain run uses seed 43; adjusted Rand index compares memberships
  without relying on cluster numbering. This is a limited seed check, not a
  full resolution/neighbor-count robustness analysis.
- GC and log10 CDS-length eta-squared report the fraction of each feature's
  variance explained by cluster membership. They measure association, not
  causality. GC/UMAP Spearman correlations characterize the displayed projection.
- GC and CDS-length boxplots, cluster sizes, mean pairwise cosine, and boundary
  weight fractions help identify composition-driven structure.

The completed embedding audit already found 97 exact duplicate vector rows
beyond the first occurrences and a median cosine of about 0.949 among 10,000
random distinct gene pairs. High raw cosine alone is therefore not strong evidence
of specific biological similarity. Exact duplicates are retained as separate
labeled genes; nearest-neighbor tie handling is documented above. No biological
cluster names or pathway enrichments are inferred automatically. Functional
interpretation should examine gene sets against the complete embedded-gene
background, alongside GC/length controls and parameter sensitivity.

## Outputs

```text
results/gb_dna_300m_cds_2026-09-28/louvain_r0.3_k30/
  analysis_manifest.json          # Completion, parameters, versions and provenance
  cosine_knn_graph.npz             # Symmetric weighted sparse graph
  graph_nodes.csv                 # Matrix row index → HGNC symbol and cluster
  gene_clusters.csv               # Every gene, cluster, centrality and metadata
  cluster_summary.csv             # Per-cluster size, composition and cohesion
  top_genes_per_cluster.csv       # Ranked top 20 (or all genes in smaller clusters)
  umap_coordinates.csv            # Coordinates used for display, keyed by symbol
  qc.json                         # Graph, silhouette, stability and confound metrics
  cluster_umap.png / .pdf         # Cluster, GC and length panels on the same UMAP
  cluster_qc.png / .pdf            # Sizes, cohesion and composition distributions
  cluster_XX_top_genes.png        # One centrality-ranked bar plot per cluster
  top_genes_all_clusters.pdf      # All representative-gene plots, one page per cluster
  report.md                       # Compact measured characterization and top genes
```

The reusable implementation is `llm_mapper/gene_clusters.py`. Its CLI takes a
vector DB from any compatible model; the GB checkpoint is not hard-coded.

The [read-retrieval experiment](na19239_read_retrieval.md) recovered and persisted
the original reference-only fit, verified identical coordinates, and regenerated
all earlier UMAP views. Future query reads use only its saved `transform`.
