"""Sequence alignment scores independent of embedding models and vector storage."""
from .smith_waterman import SWConfig, SmithWaterman, reference_sw_score

__all__ = ['SWConfig', 'SmithWaterman', 'reference_sw_score']
