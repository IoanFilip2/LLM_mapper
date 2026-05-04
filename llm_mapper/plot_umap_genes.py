import numpy as np
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# List of genes of interest
GENES_OF_INTEREST = [
    'TP53', 'KRAS', 'HLA-A', 'HLA-B', 'HLA-C',
    'EGFR', 'MYC', 'BRCA1', 'BRCA2', 'CDKN2A', 'BRAF', 'MDM2'
]

EMBED_PATH = 'data/genome/cds_gene_embeddings.npy'
META_PATH = 'data/genome/cds_gene_metadata.csv'

# Load data
print("Loading embeddings...")
embeds = np.load(EMBED_PATH, mmap_mode="r")
print("Loading metadata...")
meta = pd.read_csv(META_PATH)

# Strictly: all embeddings for genes of interest, only full-gene embedding for others (gray)
special_mask = meta['gene_symbol'].isin(GENES_OF_INTEREST)
special = meta[special_mask].copy()
other_mask = ~special_mask
other = meta[other_mask].copy()

meta_plot = pd.concat([special, other], ignore_index=True)
embeds_plot = np.concatenate([
    embeds[special.index],
    embeds[other.index]
], axis=0)

# Assign color class: gene symbol for genes of interest, else 'Other'
meta_plot['color_class'] = meta_plot['gene_symbol'].apply(lambda g: str(g).strip() if g in GENES_OF_INTEREST else 'Other')

print(meta_plot['color_class'].value_counts()) # DEBUG: see if your special genes are present
print(f"Selected {meta_plot.shape[0]} embeddings for UMAP ...")

print("Running UMAP...")
proj = umap.UMAP(random_state=42, min_dist=0.3).fit_transform(embeds_plot)
meta_plot['UMAP1'] = proj[:, 0]
meta_plot['UMAP2'] = proj[:, 1]
meta = meta_plot

# Build category for coloring (genes of interest highlighted, rest = 'Other')
def gene_cat(g):
    return g if g in GENES_OF_INTEREST else 'Other'
meta['gene_col'] = meta['gene_symbol'].apply(gene_cat)

# Seaborn color palette for genes of interest; gray for others
palette = dict(zip(GENES_OF_INTEREST, sns.color_palette('tab10', len(GENES_OF_INTEREST))))
palette['Other'] = (0.75,0.75,0.75,0.4)

plt.figure(figsize=(14, 8))

# 1. Plot 'Other' genes (gray) FIRST
sns.scatterplot(
    data=meta[meta["color_class"] == "Other"], x="UMAP1", y="UMAP2",
    color=(0.75,0.75,0.75,0.4),
    alpha=0.4,
    s=12,
    linewidth=0,
    label=None
)
# 2. Plot each gene-of-interest separately (on top)
for g in GENES_OF_INTEREST:
    df = meta[meta["color_class"] == g]
    sns.scatterplot(
        data=df, x="UMAP1", y="UMAP2",
        color=palette[g],
        alpha=0.7,
        s=17,
        linewidth=0,
        label=g
    )

# Highlight full-gene embeddings for selected genes
fulls = meta[meta['gene_symbol'].isin(GENES_OF_INTEREST)]
plt.scatter(
    fulls['UMAP1'], fulls['UMAP2'],
    facecolors='none', edgecolors='black', linewidths=2, s=180, marker='o', alpha=1, zorder=8
)

# Label full-gene dots
for _, row in fulls.iterrows():
    plt.text(
        row['UMAP1']+0.3, row['UMAP2']+0.16, row['gene_symbol'],
        fontsize=13, fontweight='bold', color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5), zorder=9
    )

# Seaborn manual legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=g,
           markerfacecolor=palette[g], markeredgewidth=0, markersize=11)
    for g in GENES_OF_INTEREST
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left',
           borderaxespad=0., fontsize=10, title="Gene")

plt.title("UMAP of k-mer and full-gene embeddings\nHighlighted: Full sequence of selected genes", fontsize=17)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig('data/genome/umap_gene_embeddings.png', dpi=200)
plt.show()
print("UMAP plot saved as data/genome/umap_gene_embeddings.png")
