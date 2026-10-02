"""Query official public archives and create reviewable sequencing manifests."""
import argparse
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from llm_mapper.preparation.discovery import discover_igsr, discover_ena, probe_fastq, verify_manifest_fastqs
    from llm_mapper.preparation.downloads import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    igsr = commands.add_parser("igsr")
    igsr.add_argument("--samples", nargs="+", required=True)
    igsr.add_argument("--strategy", choices=["WXS", "WGS"], required=True)
    igsr.add_argument("--length", type=int, choices=[100, 250], required=True)
    ena = commands.add_parser("ena")
    ena.add_argument("--query", default='tax_eq(9606) AND library_strategy="WXS" AND instrument_model="Illumina MiSeq"')
    probe = commands.add_parser("probe-fastq")
    probe.add_argument("url")
    probe.add_argument("--limit", type=int, default=1000)
    verify = commands.add_parser("verify-manifest")
    verify.add_argument("manifest")
    verify.add_argument("--limit", type=int, default=1000)
    for command in (igsr, ena, probe):
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "igsr":
        result = discover_igsr(args.samples, args.strategy, args.length, args.output)
        print(f"Selected {len(result['samples'])} samples; full-file validation remains required")
    elif args.command == "ena":
        result = discover_ena(args.query, args.output)
        print(f"Saved {len(result['records'])} candidates, not verified cohorts")
    elif args.command == "verify-manifest":
        verify_manifest_fastqs(args.manifest, args.limit)
        print("Recorded per-mate prefix evidence; full alignment QC is still required")
    else:
        write_json(args.output, probe_fastq(args.url, args.limit))


if __name__ == "__main__":
    main()
