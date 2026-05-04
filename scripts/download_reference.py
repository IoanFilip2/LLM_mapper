import os
import urllib.request
from pathlib import Path

def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.exists():
        print(f"File already exists: {dest}")
        return dest
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded to {dest}")
    return dest

def main():
    # Ensembl release 110, May 2024 (stable, easy to parse, with gene symbols)
    fasta_url = "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    gtf_url = "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz"
    out_dir = Path("data/genome/")
    fasta_gz = out_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    gtf_gz = out_dir / "Homo_sapiens.GRCh38.110.gtf.gz"

    download(fasta_url, fasta_gz)
    download(gtf_url, gtf_gz)
    print("Reference files are ready.")
    print("Next: use gunzip or your favorite tool to decompress the files if needed.")

if __name__ == "__main__":
    main()
