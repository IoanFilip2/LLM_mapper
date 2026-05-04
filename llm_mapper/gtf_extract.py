import csv
import gzip
from pathlib import Path

def extract_genes(gtf_path, output_csv):
    """
    Parse GTF and output: gene_id,gene_symbol,chrom,start,end,strand
    Ignores genes with no gene_name or gene_id attribute.
    """
    if str(gtf_path).endswith('.gz'):
        gtf_file = gzip.open(gtf_path, 'rt')
    else:
        gtf_file = open(gtf_path, 'rt')
    out_file = open(output_csv, 'w', newline='')
    writer = csv.writer(out_file)
    writer.writerow(['gene_id','gene_symbol','chrom','start','end','strand'])
    for line in gtf_file:
        if line.startswith('#'): continue
        fields = line.strip().split('\t')
        if fields[2] != 'gene': continue
        chrom,_,feature,start,end,_,strand,_,info = fields
        gid = gname = None
        for item in info.split(';'):
            item = item.strip()
            if item.startswith('gene_id'):
                gid = item.split(' ',1)[1].replace('"','')
            if item.startswith('gene_name'):
                gname = item.split(' ',1)[1].replace('"','')
        if gid and gname:
            writer.writerow([gid, gname, chrom, start, end, strand])
    gtf_file.close()
    out_file.close()
    print(f"Wrote {output_csv}")

if __name__ == "__main__":
    import sys
    gtf = sys.argv[1] if len(sys.argv) > 1 else "data/genome/Homo_sapiens.GRCh38.110.gtf.gz"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/genome/genes_hgnc.csv"
    extract_genes(gtf, out)
