# Zero-shot read retrieval: all_hgnc

935 reads; 19,217 callable gene labels.
Accuracy: 0.00%; macro sensitivity: 0.00%; top-5 accuracy: 0.75%; top-50 accuracy: 5.45% (51/935); top-100 accuracy: 7.81% (73/935); top-250 accuracy: 13.05% (122/935).

| Gene | Reads | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| TP53 | 100 | 0.000 | 1.000 | NA | 0.000 |
| KRAS | 73 | 0.000 | 1.000 | NA | 0.000 |
| HLA-A | 29 | 0.000 | 1.000 | NA | 0.000 |
| HLA-B | 100 | 0.000 | 1.000 | NA | 0.000 |
| HLA-C | 4 | 0.000 | 1.000 | NA | 0.000 |
| EGFR | 100 | 0.000 | 1.000 | NA | 0.000 |
| MYC | 100 | 0.000 | 1.000 | NA | 0.000 |
| BRCA1 | 100 | 0.000 | 1.000 | NA | 0.000 |
| BRCA2 | 100 | 0.000 | 1.000 | NA | 0.000 |
| CDKN2A | 29 | 0.000 | 1.000 | NA | 0.000 |
| BRAF | 100 | 0.000 | 1.000 | NA | 0.000 |
| MDM2 | 100 | 0.000 | 0.999 | 0.000 | 0.000 |

Specificity is conditional on the other sampled target genes, not all genomic origins.
Labels are alignment-derived, not independent biological truth. Reads are selected from the highest eligible AS tier per gene.
Both orientations are encoded; the larger cosine per candidate is used without consulting truth strand. Ties use reference database order.
Individual mates are read-level observations; sequence-identical reads and shared templates need not be independent.
Reference vectors summarize complete CDS contexts, whereas queries are 100 bp; length mismatch may affect retrieval. No model fine-tuning or label fitting.

Mean-base-Q≥30 sensitivity subset (no replacement): 0/890 correct (0.00%).

See [study methods and verification](na19239_read_retrieval.md) for selection, strand handling, fixed UMAP, and interpretation.
