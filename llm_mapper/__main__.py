"""Streaming index, mapping and benchmark CLI."""
import argparse
import csv
import json
from pathlib import Path
from .embedders import EmbeddingConfig, create_embedder
from .sequences import SequenceReader, ReadChunker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index", help="Build a reference index")
    build.add_argument("input", help="FASTA/FASTQ/BAM/CRAM or gene CSV")
    build.add_argument("index")
    build.add_argument("--backend", choices=["gb", "hf", "evo2"], default="gb")
    build.add_argument("--model", default="genbio-ai/GB.DNA-300M")
    build.add_argument("--revision", default="main")
    build.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    build.add_argument("--max-length", type=int, default=4000, help="Maximum tokens, including special tokens")
    build.add_argument("--kmer", type=int, default=0)
    build.add_argument("--layer", help="Required Evo2 hidden layer")
    build.add_argument("--loader", help="Experimental Evo2 loader module:function")
    build.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    build.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    build.add_argument("--chunk-length", type=int, default=250)
    build.add_argument("--stride", type=int)
    mapping = commands.add_parser("map", help="Map streamed reads using index configuration")
    bench = commands.add_parser("benchmark", help="Benchmark mapping on one device; run separately per device")
    for command in (mapping, bench):
        command.add_argument("input")
        command.add_argument("index")
        command.add_argument("output")
    bench.add_argument("--truth", help="CSV with read_id,label (unique IDs required)")
    for command in (build, mapping, bench):
        command.add_argument("--reference", help="Exact decoding FASTA required for CRAM input")
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        command.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from .index import build_index, VectorIndex, map_records
    if args.command == "index":
        config = EmbeddingConfig(**{k: getattr(args, k) for k in EmbeddingConfig.__dataclass_fields__})
        embedder = create_embedder(config, args.device)
        result = build_index(SequenceReader(args.input, reference=args.reference), embedder, args.index,
                             ReadChunker(args.chunk_length, args.stride), args.batch_size)
        print(json.dumps(result, indent=2))
        return
    manifest = json.loads((Path(args.index) / "manifest.json").read_text())
    embedder = create_embedder(EmbeddingConfig(**manifest["config"]), args.device)
    index = VectorIndex(args.index, embedder)
    if args.command == "benchmark":
        from .benchmark import benchmark
        truth = None
        if args.truth:
            truth = {}
            with open(args.truth) as handle:
                for row in csv.DictReader(handle):
                    if row["read_id"] in truth:
                        raise ValueError("Duplicate read ID in truth file")
                    truth[row["read_id"]] = row["label"]
        result = benchmark(args.input, index, args.batch_size, truth, reference=args.reference)
        with open(args.output, "x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    else:
        with open(args.output, "x", newline="") as handle:
            fields = ["read_id", "read_start", "read_end", "reference_id", "predicted_label",
                      "reference_start", "reference_end", "cosine_distance"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(map_records(SequenceReader(args.input, reference=args.reference), index, args.batch_size))


if __name__ == "__main__":
    main()
