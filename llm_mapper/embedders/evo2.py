"""Isolated Evo2 adapter; no global PyTorch/kernel monkey-patching.

An experimental CPU/MPS loader (module:function) must return an Evo2-compatible
wrapper with tokenizer, model and hidden-layer extraction callable.
"""
import importlib
import numpy as np
import torch
from . import SequenceEmbedder
from ..devices import resolve_device
from ..sequences import batches, SequenceRecord


class Evo2Embedder(SequenceEmbedder):
    def __init__(self, config, device="auto"):
        self.config = config
        self.device = resolve_device(device)
        if not config.layer or config.pooling != "mean" or config.kmer:
            raise ValueError("Evo2 requires an explicit layer, mean pooling, and raw nucleotides")
        if config.loader:
            module, function = config.loader.split(":", 1)
            self.wrapper = getattr(importlib.import_module(module), function)(config=config, device=self.device)
        elif self.device == "cuda":
            from evo2 import Evo2
            self.wrapper = Evo2(config.model)
        else:
            raise RuntimeError("Evo2 CPU/MPS requires a validated --loader module:function; "
                               "the evo2-silicon GUI is not a supported library loader")
        self.wrapper.model.to(device=self.device, dtype=getattr(torch, config.dtype)).eval()

    def encode(self, sequences, batch_size=8):
        if isinstance(sequences, str):
            sequences = [sequences]
        outputs = []
        # Unpadded inference avoids altering causal Hyena states.
        for batch in batches(sequences, batch_size):
            for sequence in batch:
                sequence = SequenceRecord("read", sequence).sequence
                ids = self.wrapper.tokenizer.tokenize(sequence)
                if len(ids) > self.config.max_length:
                    raise ValueError("Sequence exceeds Evo2 context; use ReadChunker")
                tokens = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    _, hidden = self.wrapper(tokens, return_embeddings=True, layer_names=[self.config.layer])
                    embedding = hidden[self.config.layer].float().mean(dim=1).cpu().numpy()
                if embedding.ndim != 2 or not np.isfinite(embedding).all():
                    raise ValueError("Invalid Evo2 embeddings")
                outputs.append(embedding)
        if not outputs:
            raise ValueError("Evo2 requires at least one sequence")
        return np.concatenate(outputs)
