"""Compatibility entry point; use `python -m llm_mapper index --help`."""
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_mapper.embedders import EmbeddingConfig, create_embedder


def get_embedder(model_name, use_kmer=True, device="auto"):
    aliases = {"aido_rna_650m_cds": "genbio-ai/AIDO.RNA-650M-CDS"}
    model_name = aliases.get(model_name.lower(), model_name)
    backend = "gb" if model_name.startswith("genbio-ai/") else "hf"
    model = create_embedder(EmbeddingConfig(
        backend=backend, model=model_name, kmer=6 if use_kmer and backend == "hf" else 0
    ), device)
    return model.encode


def main():
    from llm_mapper.__main__ import main as cli
    cli(["index", *sys.argv[1:]])


if __name__ == "__main__":
    main()
