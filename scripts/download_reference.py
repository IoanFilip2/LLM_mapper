"""Download matched GRCh38/Ensembl/HGNC sources and extract approved gene spans."""
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_mapper.preparation.downloads import Download, fetch


def download(url, dest):
    dest = Path(dest)
    return fetch(Download(url, dest.name), dest.parent)


def main():
    from llm_mapper.preparation.__main__ import main as cli
    cli(["reference", *sys.argv[1:]])


if __name__ == "__main__":
    main()
