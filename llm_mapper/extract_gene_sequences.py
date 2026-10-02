import pysam
import csv
from pathlib import Path

def reverse_complement(seq: str) -> str:
    lookup = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn", "TGCAYRKMVHDBNtgcayrkmvhdbn")
    return seq.translate(lookup)[::-1]

def get_gene_sequences(fasta_path, gene_csv, output_csv):
    fasta = pysam.FastaFile(str(fasta_path))
    with open(gene_csv) as incsv, open(output_csv, 'w', newline='') as outcsv:
        reader = csv.DictReader(incsv)
        fieldnames = reader.fieldnames + ['sequence']
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            chrom = row['chrom']
            start = int(row['start']) - 1  # GTF is 1-based, python 0-based
            end = int(row['end'])
            strand = row.get('strand', '+')
            try:
                seq = fasta.fetch(chrom, start, end)
            except Exception as e:
                print(f"Warning: Could not fetch {chrom}:{start+1}-{end}: {e}")
                seq = ''
            if strand == '-':
                seq = reverse_complement(seq)
            row['sequence'] = seq
            writer.writerow(row)
    print(f"Saved gene sequences to {output_csv}")

if __name__ == "__main__":
    import sys
    fasta = sys.argv[1] if len(sys.argv) > 1 else "data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    gene_csv = sys.argv[2] if len(sys.argv) > 2 else "data/genome/genes_hgnc.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "data/genome/gene_sequences.csv"
    get_gene_sequences(fasta, gene_csv, output_csv)
