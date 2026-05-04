from Bio import SeqIO
import pandas as pd

fasta_path = "data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
gtf_path = "data/genome/genes_hgnc.csv"  # Assumes contains parsed GTF info

print("Chromosomes in FASTA:")
print(sorted(set(rec.id for rec in SeqIO.parse(fasta_path, "fasta"))))
gtf_chroms = pd.read_csv(gtf_path)['chrom'].unique()
print("\nChromosomes in GTF:")
print(sorted(set(gtf_chroms)))
