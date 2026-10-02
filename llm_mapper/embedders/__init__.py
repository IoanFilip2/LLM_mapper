"""Model-independent embedding contract; optional backends load on demand."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "gb"
    model: str = "genbio-ai/GB.DNA-300M"
    revision: str = "main"
    pooling: str = "mean"
    max_length: int = 4000
    kmer: int = 0
    layer: str | None = None
    loader: str | None = None
    dtype: str = "float32"
    attention: str = "eager"

    def __post_init__(self):
        if self.backend not in {"gb", "hf", "evo2"}:
            raise ValueError("Backend must be gb, hf, or evo2")
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("Pooling must be mean or cls")
        if self.max_length < 1 or self.kmer < 0:
            raise ValueError("Invalid context length or k-mer size")
        if self.attention not in {"eager", "sdpa"}:
            raise ValueError("Attention must be eager or sdpa")
        if self.attention == "sdpa" and self.backend != "gb":
            raise ValueError("Explicit SDPA adapter currently supports GB only")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")


class SequenceEmbedder(ABC):
    """encode returns finite float32 [sequence, feature] vectors, in input order."""
    config: EmbeddingConfig
    device: str

    @property
    def signature(self):
        return {"api_version": 2, **asdict(self.config)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()

    @abstractmethod
    def encode(self, sequences, batch_size=8):
        pass


def create_embedder(config=None, device="auto"):
    config = config or EmbeddingConfig()
    if config.backend == "evo2":
        from .evo2 import Evo2Embedder
        return Evo2Embedder(config, device)
    from .transformer import TransformerEmbedder
    return TransformerEmbedder(config, device)
