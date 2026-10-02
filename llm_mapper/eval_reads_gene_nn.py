"""Compatibility entry point; use `python -m llm_mapper benchmark --help`."""
import sys
from pathlib import Path
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from llm_mapper.__main__ import main as cli
    cli(["benchmark", *sys.argv[1:]])


if __name__ == "__main__":
    main()
