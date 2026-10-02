# Zero-shot read retrieval: targets_only

935 reads; 12 callable gene labels.
Accuracy: 32.83%; macro sensitivity: 31.20%; top-5 accuracy: 86.52%; top-50 accuracy: 100.00% (935/935); top-100 accuracy: 100.00% (935/935); top-250 accuracy: 100.00% (935/935).

| Gene | Reads | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| TP53 | 100 | 0.920 | 0.674 | 0.253 | 0.397 |
| KRAS | 73 | 0.685 | 0.886 | 0.338 | 0.452 |
| HLA-A | 29 | 0.000 | 0.991 | 0.000 | 0.000 |
| HLA-B | 100 | 0.010 | 0.925 | 0.016 | 0.012 |
| HLA-C | 4 | 0.000 | 0.967 | 0.000 | 0.000 |
| EGFR | 100 | 0.050 | 0.976 | 0.200 | 0.080 |
| MYC | 100 | 0.310 | 0.996 | 0.912 | 0.463 |
| BRCA1 | 100 | 0.300 | 0.940 | 0.375 | 0.333 |
| BRCA2 | 100 | 0.410 | 0.980 | 0.707 | 0.519 |
| CDKN2A | 29 | 0.690 | 0.992 | 0.741 | 0.714 |
| BRAF | 100 | 0.000 | 0.994 | 0.000 | 0.000 |
| MDM2 | 100 | 0.370 | 0.935 | 0.407 | 0.387 |

Specificity is conditional on the other sampled target genes, not all genomic origins.
Labels are alignment-derived, not independent biological truth. Reads are selected from the highest eligible AS tier per gene.
Both orientations are encoded; the larger cosine per candidate is used without consulting truth strand. Ties use reference database order.
Individual mates are read-level observations; sequence-identical reads and shared templates need not be independent.
Reference vectors summarize complete CDS contexts, whereas queries are 100 bp; length mismatch may affect retrieval. No model fine-tuning or label fitting.

Mean-base-Q≥30 sensitivity subset (no replacement): 294/890 correct (33.03%).

See [study methods and verification](na19239_read_retrieval.md) for selection, strand handling, fixed UMAP, and interpretation.

Top-50, top-100 and top-250 each include all 12 candidates in this restricted universe, so 100% is guaranteed and is not evidence of retrieval quality.
