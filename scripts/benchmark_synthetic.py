"""Reproducible runtime smoke benchmark, NOT a biological accuracy evaluation.

Run CPU first to build the reference index, then MPS with the same work directory:
  python -m scripts.benchmark_synthetic --device cpu --work-dir data/smoke --output cpu.json
  python -m scripts.benchmark_synthetic --device mps --work-dir data/smoke --output mps.json
"""
import argparse
import json
from pathlib import Path
import random
from llm_mapper.embedders import EmbeddingConfig, create_embedder
from llm_mapper.index import build_index, VectorIndex
from llm_mapper.sequences import SequenceReader, ReadChunker
from llm_mapper.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    references = ["".join(rng.choices("ACGT", k=250)) for _ in range(4)]
    fasta = args.work_dir / "references.fa"
    fasta.write_text("".join(f">gene{i}\n{s}\n" for i, s in enumerate(references)))
    index_path = args.work_dir / "index"
    config = (EmbeddingConfig(**json.loads((index_path / "manifest.json").read_text())["config"])
              if index_path.exists() else EmbeddingConfig())
    embedder = create_embedder(config, args.device)
    if not index_path.exists():
        build_index(SequenceReader(fasta), embedder, index_path, ReadChunker(250), batch_size=4)
    index = VectorIndex(index_path, embedder)
    reports = []
    for length in [100, 150, 250]:
        path = args.work_dir / f"reads-{length}.fq"
        truth = {}
        with path.open("w") as handle:
            for i, seq in enumerate(references):
                for offset in [0, (250 - length) // 2, 250 - length]:
                    name = f"gene{i}-read{len(truth)}"
                    truth[name] = f"gene{i}"
                    handle.write(f"@{name}\n{seq[offset:offset+length]}\n+\n{'I'*length}\n")
        for repeat in range(3):
            report = benchmark(path, index, batch_size=4, truth=truth)
            reports.append({"read_length": length, "repeat": repeat, **report})
    result = {"purpose": "Synthetic runtime smoke test; exact substrings of random references. "
                         "Accuracy is not evidence of held-out biological mapping performance.",
              "seed": 42, "signature": embedder.signature, "reports": reports}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
