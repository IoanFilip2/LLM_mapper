"""Download selected reviewed WES samples; this is no longer a chromosome demo.

A full exome is not a small test BAM. Use --sample to request one sample, or use
unit-test fixtures for tiny offline tests. Original sources are retained.
"""
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from llm_mapper.preparation.__main__ import main as cli
    cli(["samples", "configs/wes_100bp_1000genomes.json", *sys.argv[1:]])


if __name__ == "__main__":
    main()
