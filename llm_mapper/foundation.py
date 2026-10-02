"""Compatibility facade for the original FoundationModel API."""
from llm_mapper.embedders import EmbeddingConfig, create_embedder


class FoundationModel:
    def __init__(self, model_name, device=None, use_kmer=True):
        self.model_name = model_name
        self.use_kmer = use_kmer
        backend = "gb" if model_name.startswith("genbio-ai/") else "hf"
        self._embedder = create_embedder(EmbeddingConfig(
            backend=backend, model=model_name, pooling="cls", kmer=6 if use_kmer else 0
        ), device=device or "auto")
        self.device = self._embedder.device
        self.model = self._embedder.model
        self.tokenizer = self._embedder.tokenizer

    @staticmethod
    def kmerize(seq, k=6):
        seq = "".join(seq.split()).upper()
        return " ".join(seq[i:i+k] for i in range(len(seq)-k+1))

    def encode(self, sequences, batch_size=8, k=6, verbose=False):
        if self.use_kmer and k != self._embedder.config.kmer:
            raise ValueError("K-mer size is fixed in the embedding configuration")
        return self._embedder.encode(sequences, batch_size=batch_size)
