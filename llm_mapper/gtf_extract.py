"""Legacy generic GTF gene coordinates; HGNC filtering lives in preparation.reference."""
import csv
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def extract_genes(gtf_path, output_csv):
    from llm_mapper.preparation.reference import gtf_records
    with open(output_csv, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["gene_id", "gene_symbol", "chrom", "start", "end", "strand"])
        for gene in gtf_records(gtf_path):
            if gene.get("gene_id") and gene.get("gene_name"):
                writer.writerow([gene["gene_id"], gene["gene_name"], gene["chrom"],
                                 gene["start"], gene["end"], gene["strand"]])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gtf")
    parser.add_argument("output")
    args = parser.parse_args()
    extract_genes(args.gtf, args.output)
