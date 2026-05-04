import pandas as pd
import gzip
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
import csv

FASTA_PATH = "data/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
GTF_PATH = "data/genome/Homo_sapiens.GRCh38.110.gtf"
OUT_CSV = "data/genome/gene_cds_sequences.csv"

# --- 1. Index genome sequence ---
print("Indexing FASTA...")
chrom_seqs = SeqIO.to_dict(SeqIO.parse(FASTA_PATH, "fasta"))

# --- 2. Parse GTF for CDS exons ---
print("Extracting CDS from GTF...")
gtf_open = gzip.open if str(GTF_PATH).endswith(
    ".gz") else open

# Map gene_id -> list of (chrom, start, end, strand)
gene_cds_map = defaultdict(list)
gene_symbol_map = {}
with gtf_open(GTF_PATH, "rt") as gtf:
    for line in gtf:
        if line.startswith('#'): continue
        fields = line.strip().split('\t')
        chrom, _, feature, start, end, _, strand, _, info = fields
        if feature != 'CDS': continue
        info_dict = {i.split()[0]: i.split()[1].replace('"','') for i in info.strip().split(';') if i.strip()}
        gene_id = info_dict.get('gene_id')
        gene_symbol = info_dict.get('gene_name', '')
        if gene_id:
            gene_cds_map[gene_id].append((chrom, int(start), int(end), strand))
            if gene_symbol:
                gene_symbol_map[gene_id] = gene_symbol

print(f"Found {len(gene_cds_map)} genes with CDS.")

# --- 3. Extract and concatenate CDS sequence for each gene ---
with open(OUT_CSV, "w", newline="") as outcsv:
    fieldnames = ["gene_id", "gene_symbol", "chrom", "strand", "cds_parts", "sequence"]
    writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
    writer.writeheader()
    for gene_id, cds_parts in gene_cds_map.items():
        # sort by genomic order (for + strand) or reverse order (for - strand)
        strand = cds_parts[0][3]
        chrom = cds_parts[0][0]
        if strand == '+':
            cds_parts = sorted(cds_parts, key=lambda x: x[1])
        else:
            cds_parts = sorted(cds_parts, key=lambda x: x[1], reverse=True)
        fullseq = ""
        cds_coords = []
        for chrom, start, end, strand in cds_parts:
            seqfrag = chrom_seqs[chrom][start-1:end].seq
            fullseq += seqfrag
            cds_coords.append(f"{start}-{end}")
        if strand == '-':
            from Bio.Seq import Seq as BioSeq
            fullseq = str(BioSeq(fullseq).reverse_complement())
        writer.writerow({
            "gene_id": gene_id,
            "gene_symbol": gene_symbol_map.get(gene_id, ''),
            "chrom": chrom,
            "strand": strand,
            "cds_parts": ','.join(cds_coords),
            "sequence": str(fullseq)
        })

print(f"Saved CDS-based gene sequences to {OUT_CSV}")
