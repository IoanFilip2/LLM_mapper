import numpy as np
import glob
import os
from tqdm import tqdm

BATCH_DIR = "data/genome/full_gene_embedding_batches"
OUT_NPY = "data/genome/cds_gene_embeddings.npy"

# 1. Get sorted batch files
batches = sorted(
    glob.glob(f"{BATCH_DIR}/embeddings_batch_*.npy"),
    key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
)

# Guard if no batches found
if not batches:
    raise RuntimeError(f"No batch files matched in {BATCH_DIR}/embeddings_batch_*.npy")

# First pass: sum all rows for preallocation
total_rows = sum(np.load(batch, mmap_mode="r").shape[0] for batch in batches)
EMBED_DIM = 768  # adjust if your embeddings are different
print(f"Preparing output array of shape: ({total_rows}, {EMBED_DIM})")

# Pre-allocate output file as memmap
merged = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype="float32", shape=(total_rows, EMBED_DIM))

curr_idx = 0
for i, batch in enumerate(tqdm(batches, desc="Merging batches", unit="batch")):
    arr = np.load(batch)
    rows = arr.shape[0]
    merged[curr_idx:curr_idx+rows] = arr
    merged.flush()
    curr_idx += rows
    # Print growing file size in MB
    size_mb = os.path.getsize(OUT_NPY) / (1024 * 1024)
    print(f"Batch {i+1}/{len(batches)}: {rows} rows, merged file size: {size_mb:.1f} MB", end='\r', flush=True)

print()
print(f"Done. Final merged embeddings: {OUT_NPY} (rows: {total_rows}, dim: {EMBED_DIM})")
final_size_mb = os.path.getsize(OUT_NPY) / (1024 * 1024)
print(f"Final merged file size: {final_size_mb:.1f} MB")