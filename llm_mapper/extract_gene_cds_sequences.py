"""Compatibility entry point for transcript-aware, strand-correct CDS extraction.

Run `python -m llm_mapper.extract_gene_cds_sequences --help` for explicit inputs.
The old concatenation of all transcript CDS records has been replaced.
"""
if __package__ in {None, ''}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_mapper.preparation.cds import main

if __name__ == '__main__':
    main()
