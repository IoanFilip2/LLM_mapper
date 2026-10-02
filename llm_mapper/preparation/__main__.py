"""Prepare references and audited public sequencing inputs independently of models."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create clearly separated reference/WES/WGS data directories")
    init.add_argument("--root", default="data")
    ref = commands.add_parser("reference", help="Download matched Ensembl/HGNC files and extract full gene spans")
    ref.add_argument("--root", default="data/reference")
    ref.add_argument("--release", type=int, default=110)
    ref.add_argument("--source-manifest", help="Optional reference checksum lock; mutable HGNC URLs must match it")
    ref.add_argument("--sequence-csv", action="store_true", help="Also include sequence strings in genes.csv")
    extract = commands.add_parser("extract", help="Extract from existing local inputs")
    for name in ("fasta", "gtf", "hgnc", "output"):
        extract.add_argument("--" + name, required=True)
    extract.add_argument("--sequence-csv", action="store_true")
    samples = commands.add_parser("samples", help="Download and QC a reviewed sample manifest directly")
    samples.add_argument("manifest")
    samples.add_argument("--root", default="data")
    samples.add_argument("--sample", action="append", help="Restrict to listed sample IDs")
    samples.add_argument("--download-only", action="store_true")
    samples.add_argument("--export-bam", action="store_true", help="Optional additional BAM derivative; QC uses source alignment")
    inspect = commands.add_parser("inspect", help="Measure untrimmed primary read lengths and assembly compatibility")
    inspect.add_argument("alignment")
    inspect.add_argument("--reference")
    inspect.add_argument("--expected-length", type=int, required=True)
    inspect.add_argument("--limit", type=int, default=0, help="0 scans all records; positive values are prefix QC only")
    inspect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        from .layout import initialize
        print(json.dumps(initialize(args.root), indent=2))
    elif args.command == "reference":
        from .reference import prepare_reference
        result = prepare_reference(args.root, args.release, args.sequence_csv, args.source_manifest)
        print(json.dumps({k: result[k] for k in ("genes_written", "hgnc_approved_protein_coding", "total_gene_bases")}, indent=2))
    elif args.command == "extract":
        from .reference import extract_hgnc_reference
        result = extract_hgnc_reference(args.fasta, args.gtf, args.hgnc, args.output, args.sequence_csv)
        print(json.dumps({"genes_written": result["genes_written"], "unresolved": len(result["unresolved"])}, indent=2))
    elif args.command == "samples":
        from .samples import prepare_samples
        prepare_samples(args.manifest, args.root, args.sample, args.download_only, args.export_bam)
    else:
        from .samples import inspect_alignment
        from .downloads import write_json
        write_json(args.output, inspect_alignment(args.alignment, args.expected_length, args.reference, args.limit))


if __name__ == "__main__":
    main()
