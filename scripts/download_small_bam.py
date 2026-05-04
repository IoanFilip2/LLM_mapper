import os
from urllib.request import urlretrieve
from pathlib import Path

# GIAB chr6 NA12878 high-coverage BAM (trimmed, <200MB)
# Source: https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/10XGenomics/latest/NA24385_pbjelly_bam/
# We'll use a chr6-specific slice (test region)

# For testing, use linked test file from ENSEMBL: chr6 of NA12878 1000 Genomes (public tiny BAM for demo)
BAM_URL = "https://ftp.ensembl.org/pub/grch37/current/bam/NA12878.chrom6.SLX.maq.SRP000032.2009_07.bam"
BAI_URL = "https://ftp.ensembl.org/pub/grch37/current/bam/NA12878.chrom6.SLX.maq.SRP000032.2009_07.bam.bai"

out_dir = Path("data/test_bam")
out_dir.mkdir(exist_ok=True, parents=True)
bam_path = out_dir / "NA12878.chrom6.SLX.maq.SRP000032.2009_07.bam"
bai_path = out_dir / "NA12878.chrom6.SLX.maq.SRP000032.2009_07.bam.bai"

def download(url, dest):
    if dest.exists():
        print(f"File exists: {dest}")
    else:
        print(f"Downloading {url} -> {dest} ...")
        urlretrieve(url, dest)
        print(f"Done: {dest}")

if __name__ == "__main__":
    download(BAM_URL, bam_path)
    download(BAI_URL, bai_path)
    print("BAM and BAI ready for use!")
