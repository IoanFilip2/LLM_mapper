import numpy as np
import pandas as pd
from collections import defaultdict

meta_path = "data/genome/gene_kmer_metadata.csv"
embed_path = "data/genome/gene_kmer_embeddings.npy"

print("Loading metadata and full-gene embeddings...")
meta = pd.read_csv(meta_path)
mask = meta['is_full_gene'] == 1
gene_indices = np.flatnonzero(mask.values)
n_total_full = len(gene_indices)

# Randomly subsample 1000
import random
random.seed(42)
subsample_idx = sorted(random.sample(list(gene_indices), min(1000, n_total_full)))
gene_embeds = np.load(embed_path, mmap_mode='r')[subsample_idx]
gene_symbols = meta.iloc[subsample_idx]['gene_symbol'].values

embeddings_as_tuple = [tuple(np.round(x, 6)) for x in gene_embeds]
unique_set = set(embeddings_as_tuple)
num_unique = len(unique_set)
num_total = len(gene_embeds)
num_dupe = num_total - num_unique

print(f"Subsampled {num_total} full-gene embeddings.")
print(f"Unique in subsample: {num_unique}")
print(f"Duplicated in subsample: {num_dupe}")

unique_vectors = list({tup for tup in embeddings_as_tuple})
if unique_vectors:
    print("The unique embedding vector (rounded to 6 decimals, first 20 dims shown):")
    print(np.array(unique_vectors[0])[:20])
    print("Full vector (all dimensions, rounded to 6 decimals):")
    print(np.array(unique_vectors[0]))

# Optionally, show examples of duplicates

dupemap = defaultdict(list)
for i, tup in enumerate(embeddings_as_tuple):
    dupemap[tup].append(gene_symbols[i])
dupes = [(genes, len(genes)) for genes in dupemap.values() if len(genes) > 1]
if dupes:
    print("Example duplicate embedding for different genes:")
    for genes, n in dupes[:5]:
        print(f"  {n} genes: {genes}")
else:
    print("No duplicated full-gene embeddings found in this subsample.")
