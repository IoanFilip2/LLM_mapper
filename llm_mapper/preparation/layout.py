"""Stable input locations; model-specific artifacts belong elsewhere."""
from pathlib import Path

DIRECTORIES = {
    "reference": "GRCh38, matching annotation, HGNC snapshot, full gene sequences, and provenance.",
    "catalog": "Archive metadata snapshots and discovery evidence; not sequencing reads.",
    "wes_100bp_1000genomes": "1000 Genomes WES with native 100-bp libraries. Full-file QC required before use.",
    "wes_250bp_public": "Reserved for verified native 250-bp WES. No qualifying sources selected yet.",
    "wgs_250bp_comparison": "Separately approved 250-bp WGS comparison cohort. These samples are NOT WES.",
}


def initialize(root="data"):
    root = Path(root)
    for name, description in DIRECTORIES.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(f"# {name}\n\n{description}\n")
    return {name: str(root / name) for name in DIRECTORIES}
