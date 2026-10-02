"""Compatibility entry point for labeled, reproducible gene-vector UMAPs.

Example: python -m llm_mapper.plot_umap_genes results/RUN/vector_db results/RUN/plots
"""
if __package__ in {None, ''}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_mapper.gene_umap import GENES_OF_INTEREST, main

if __name__ == '__main__':
    main()
