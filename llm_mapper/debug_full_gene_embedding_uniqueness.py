import numpy as np
import pandas as pd

meta_path = "data/genome/gene_kmer_metadata.csv"
embed_path = "data/genome/gene_kmer_embeddings.npy"

meta = pd.read_csv(meta_path)
mask = meta['is_full_gene'] == 1
gene_embeds = np.load(embed_path, mmap_mode='r')[mask.values]
gene_symbols = meta.loc[mask, 'gene_symbol'].values

inds = np.random.choice(len(gene_embeds), size=5, replace=False)
for i in inds:
    print(f"\nGene: {gene_symbols[i]}")
    print(f"Sequence len: {len(meta.loc[mask].iloc[i]['sequence'])}")
    print("Embedding (first 10 dims):", np.round(gene_embeds[i][:10], 6))

print("\nPairwise diff among these 5 full-gene vectors:")
for i in range(len(inds)):
    for j in range(i+1, len(inds)):
        diff = np.linalg.norm(gene_embeds[inds[i]] - gene_embeds[inds[j]])
        print(f"{gene_symbols[inds[i]]} vs {gene_symbols[inds[j]]}: ||diff|| = {diff:.6f}")
