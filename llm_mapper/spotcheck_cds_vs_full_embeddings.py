import numpy as np
import pandas as pd

full_meta = pd.read_csv("data/genome/full_gene_metadata.csv")
cds_meta  = pd.read_csv("data/genome/cds_gene_metadata.csv")
full_embeds = np.load("data/genome/full_gene_embeddings.npy", mmap_mode="r")
cds_embeds  = np.load("data/genome/cds_gene_embeddings.npy", mmap_mode="r")

genes_to_check = ["HLA-A", "KRAS", "CDKN2A"]  # add more as needed

for gene in genes_to_check:
    i_full = full_meta.index[full_meta['gene_symbol'] == gene]
    i_cds  = cds_meta.index[cds_meta['gene_symbol'] == gene]
    if not len(i_full) or not len(i_cds):
        print(f"{gene}: not found in one or both datasets")
        continue
    v_full = full_embeds[i_full[0]]
    v_cds  = cds_embeds[i_cds[0]]
    diff = np.linalg.norm(v_full - v_cds)
    print(f"\nGene {gene}:")
    print("  Full-genome embedding (first 10):", np.round(v_full[:10], 4))
    print("  CDS-only embedding   (first 10):", np.round(v_cds[:10], 4))
    print(f"  L2 distance between embeddings: {diff:.4f}")

# Additionally, spot-check that all rows are unique within each matrix as well
print(f"\n# unique full-gene embeddings: {len(set([tuple(np.round(x, 6)) for x in full_embeds]))} / {len(full_embeds)}")
print(f"# unique CDS-gene embeddings : {len(set([tuple(np.round(x, 6)) for x in cds_embeds]))} / {len(cds_embeds)}")
