"""GB.DNA/GB.RNA and Hugging Face encoders with masked sequence pooling."""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from . import SequenceEmbedder
from ..devices import resolve_device
from ..sequences import batches, SequenceRecord


def pool_hidden(hidden, attention_mask, special_mask, pooling):
    if hidden.ndim != 3:
        raise ValueError("Expected token embeddings [batch, token, feature]")
    if pooling == "cls":
        return hidden[:, 0].float()
    mask = attention_mask.bool() & ~special_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("No nucleotide tokens to pool")
    weights = mask.unsqueeze(-1).float()
    return (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1)


class TransformerEmbedder(SequenceEmbedder):
    def __init__(self, config, device="auto"):
        self.device = resolve_device(device)
        self.config = config
        if config.backend == "gb":
            from modelgenerator.huggingface_models.rnabert.modeling_rnabert import RNABertModel
            from modelgenerator.huggingface_models.rnabert.tokenization_rnabert import RNABertTokenizer
            model_class, tokenizer_class = RNABertModel, RNABertTokenizer
            import modelgenerator.huggingface_models.rnabert as rnabert
            local_vocab = Path(config.model) / "vocab.txt"
            vocab = local_vocab if local_vocab.is_file() else Path(rnabert.__file__).parent / "vocab.txt"
            self.tokenizer = tokenizer_class(str(vocab), version="v2")
        else:
            from transformers import AutoModel, AutoTokenizer
            model_class, tokenizer_class = AutoModel, AutoTokenizer
        self.model, loading = model_class.from_pretrained(
            config.model, revision=config.revision, output_loading_info=True
        )
        missing = [key for key in loading["missing_keys"] if not key.startswith("pooler.")]
        if missing or loading.get("mismatched_keys"):
            raise ValueError(f"Checkpoint is missing encoder weights: {missing}")
        resolved = getattr(self.model.config, "_commit_hash", None)
        if resolved:
            self.config = replace(config, revision=resolved)
        if config.backend != "gb":
            self.tokenizer = tokenizer_class.from_pretrained(config.model, revision=self.config.revision)
        if config.attention == "sdpa":
            from .gb_attention import install_gb_sdpa
            install_gb_sdpa(self.model)
        self.model.to(device=self.device, dtype=getattr(torch, config.dtype)).eval()
        self.dimension = self.model.config.hidden_size
        self.context_length = min(config.max_length,
                                  getattr(self.model.config, "max_position_embeddings", config.max_length))

    @property
    def signature(self):
        packages = ["torch", "transformers"]
        if self.config.backend == "gb":
            packages.append("modelgenerator")
        vocab_hash = hashlib.sha256(json.dumps(self.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
        return {**super().signature, "vocabulary_sha256": vocab_hash,
                "packages": {name: version(name) for name in packages}}

    def encode(self, sequences, batch_size=8):
        if isinstance(sequences, str):
            sequences = [sequences]
        outputs = []
        for batch in batches(sequences, batch_size):
            seqs = [SequenceRecord(str(i), s).sequence for i, s in enumerate(batch)]
            if self.config.kmer:
                k = self.config.kmer
                if any(len(s) < k for s in seqs):
                    raise ValueError("Sequence shorter than configured k-mer")
                seqs = [" ".join(s[i:i+k] for i in range(len(s)-k+1)) for s in seqs]
            tokens = self.tokenizer(seqs, padding=True, truncation=False,
                                    return_special_tokens_mask=True, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.context_length:
                raise ValueError(f"Tokenized sequence exceeds {self.context_length} tokens; use ReadChunker")
            special = tokens.pop("special_tokens_mask").to(self.device)
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                pooled = pool_hidden(hidden, tokens["attention_mask"], special, self.config.pooling)
            array = pooled.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Model returned non-finite embeddings")
            outputs.append(array)
        return np.concatenate(outputs) if outputs else np.empty((0, self.dimension), dtype=np.float32)
