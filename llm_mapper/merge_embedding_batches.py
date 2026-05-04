import numpy as np
import glob
import os

BATCH_DIR = "data/genome/gene_kmer_embedding_batches"
OUT_NPY = "data/genome/gene_kmer_embeddings.npy"

batches = sorted(
    glob.glob(f"{BATCH_DIR}/embeddings_batch_*.npy"),
    key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
)
print(f"Found {len(batches)} files.")
arrays = [np.load(batch) for batch in batches]
print("Shapes:", [a.shape for a in arrays])
all_embeddings = np.concatenate(arrays, axis=0)
np.save(OUT_NPY, all_embeddings)
print(f"Merged embeddings: {OUT_NPY}, shape {all_embeddings.shape}")
