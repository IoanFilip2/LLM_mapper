"""Disk-backed exact cosine index with bounded reference/query batches."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .sequences import batches, ReadChunker


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero embeddings cannot be indexed by cosine similarity")
    return vectors / norms


def build_index(records, embedder, destination, chunker=None, batch_size=8):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Index already exists: {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunker = chunker or ReadChunker()
    staging = Path(tempfile.mkdtemp(prefix=".index-", dir=destination.parent))
    count, dimension, shards = 0, None, 0
    try:
        for shard, batch in enumerate(batches(chunker(records), batch_size)):
            vectors = normalize(embedder.encode([r.sequence for r in batch], batch_size=batch_size))
            if vectors.shape[0] != len(batch) or vectors.shape[1] == 0:
                raise ValueError("Embedder returned an invalid batch shape")
            if dimension is not None and dimension != vectors.shape[1]:
                raise ValueError("Embedding dimension changed during indexing")
            dimension = vectors.shape[1]
            np.save(staging / f"{shard}.npy", vectors, allow_pickle=False)
            with (staging / f"{shard}.jsonl").open("w") as out:
                for r in batch:
                    out.write(json.dumps({"id": r.id, "label": r.label or r.id,
                                          "start": r.start, "end": r.end}) + "\n")
            count += len(batch)
            shards += 1
        if not count:
            raise ValueError("Cannot build an empty index")
        manifest = {"format_version": 1, "signature": embedder.signature,
                    "fingerprint": embedder.fingerprint, "config": asdict(embedder.config),
                    "chunker": asdict(chunker), "count": count, "dimension": dimension,
                    "shards": shards, "metric": "cosine", "device": embedder.device}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


class VectorIndex:
    def __init__(self, path, embedder):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported index version")
        if self.manifest["fingerprint"] != embedder.fingerprint:
            raise ValueError("Reference/read embedding configuration mismatch; rebuild the index")
        self.embedder = embedder

    def search(self, sequences, batch_size=8):
        queries = normalize(self.embedder.encode(sequences, batch_size=batch_size))
        if queries.shape[1] != self.manifest["dimension"]:
            raise ValueError("Reference/read embedding dimension mismatch")
        scores = np.full(len(queries), -np.inf, dtype=np.float32)
        matches = [None] * len(queries)
        for shard in range(self.manifest["shards"]):
            vectors = np.load(self.path / f"{shard}.npy", mmap_mode="r", allow_pickle=False)
            similarities = queries @ vectors.T
            indices = similarities.argmax(axis=1)
            candidate_scores = similarities[np.arange(len(queries)), indices]
            improved = candidate_scores > scores
            if improved.any():
                with (self.path / f"{shard}.jsonl").open() as handle:
                    metadata = [json.loads(line) for line in handle]
                for i in np.flatnonzero(improved):
                    scores[i] = candidate_scores[i]
                    matches[i] = metadata[indices[i]]
        return [{**match, "cosine_distance": float(1 - np.clip(score, -1, 1))}
                for match, score in zip(matches, scores)]


def map_records(records, index, batch_size=8):
    # Reference and reads share chunking and model configuration.
    chunker = ReadChunker(**index.manifest["chunker"])
    for batch in batches(chunker(records), batch_size):
        matches = index.search([r.sequence for r in batch], batch_size=batch_size)
        for record, match in zip(batch, matches):
            yield {"read_id": record.id, "read_start": record.start, "read_end": record.end,
                   "reference_id": match["id"], "predicted_label": match["label"],
                   "reference_start": match["start"], "reference_end": match["end"],
                   "cosine_distance": match["cosine_distance"]}
