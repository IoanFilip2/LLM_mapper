import pysam
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
from tqdm import tqdm
import sys
sys.path.insert(0, str(__file__).replace('/eval_reads_gene_nn.py', ''))
from foundation import FoundationModel

# ---- USER INPUT ----
BAM_PATH = "data/test_bam/HGDP00967.star.GRCh38.Aligned.sortedByCoord.bam" # update if your file has a different path or name
GENE_INFO_CSV = "data/genome/cds_gene_metadata.csv"
GENE_EMBED_NPY = "data/genome/cds_gene_embeddings.npy"
GENES = {
    "HLA-A": ("chr6", 29941260, 29949572), 
    "KRAS": ("chr12", 25204789, 25250936),  
    "CDKN2A": ("chr9", 21967752, 21995301),
}
N_READS = 100

# ---- 1. LOAD FULL-GENE EMBEDDINGS ----
print("Loading gene embeddings...")
gene_meta = pd.read_csv(GENE_INFO_CSV)
all_embeds = np.load(GENE_EMBED_NPY, mmap_mode="r")
gene_labels = gene_meta['gene_symbol'].values

# Restrict reference DB to just the target genes
TARGET_GENES = set(GENES.keys())
genemask3 = np.isin(gene_labels, list(TARGET_GENES))
gene_embeds3 = all_embeds[genemask3]
gene_labels3 = gene_labels[genemask3]


# ---- 2. EXTRACT READS BY GENE ----
print("Loading BAM and extracting reads...")
reads = []
labels = []
samtools_chr_map = {str(i): f"chr{i}" for i in range(1, 23)}
samtools_chr_map.update({"X": "chrX", "Y": "chrY", "MT": "chrM"})

# Only open the big BAM if needed
need_big_bam = False
for gene in GENES.keys():
    test_bam_path = f"data/test_bam/test_reads_{gene}.bam"
    test_bam_idx = test_bam_path + ".bai"
    if not (os.path.exists(test_bam_path) and os.path.exists(test_bam_idx)):
        need_big_bam = True
        break

bam = None
if need_big_bam:
    import pysam
    bam = pysam.AlignmentFile(BAM_PATH, "rb")

for gene, (chrom, start, end) in GENES.items():
    test_bam_path = f"data/test_bam/test_reads_{gene}.bam"
    test_bam_idx = test_bam_path + ".bai"
    unique_sequences = set()
    # Check if bam already exists
    unique_records = []
    if os.path.exists(test_bam_path) and os.path.exists(test_bam_idx):
        print(f"Loading previously extracted reads from {test_bam_path}")
        with pysam.AlignmentFile(test_bam_path, "rb") as tbam:
            for aln in tbam.fetch():
                seq = aln.query_sequence
                if seq and seq not in unique_sequences:
                    reads.extend([seq[:150]])
                    labels.extend([gene])
                    unique_sequences.add(seq)
                if len(unique_sequences) >= N_READS:
                    break
        print(f"{gene}: loaded {len(unique_sequences)} unique reads from cache BAM.")
    else:
        chroms = [chrom, samtools_chr_map.get(chrom, chrom)] # Try plain and chr-prefixed
        seen_reads = set()
        reads_gene = []
        to_write = []
        for c in chroms:
            for aln in bam.fetch(c, start, end):
                if not aln.is_unmapped and aln.mapping_quality >= 60 and not aln.is_secondary and not aln.is_supplementary:
                    seq = aln.query_sequence
                    if seq and len(seq) >= 100 and seq not in seen_reads:
                        reads_gene.append(seq[:150])
                        seen_reads.add(seq)
                        to_write.append(aln)
                if len(reads_gene) >= N_READS:
                    break
            if len(reads_gene) >= N_READS:
                break
        print(f"{gene}: found {len(reads_gene)} unique mapped reads (MAPQ>=60). Writing to BAM ...")
        # Write to BAM
        if reads_gene:
            with pysam.AlignmentFile(test_bam_path, "wb", template=bam) as outbam:
                for aln in to_write:
                    outbam.write(aln)
            pysam.index(test_bam_path)
        reads.extend(reads_gene)
        labels.extend([gene] * len(reads_gene))

print(f"Total reads found: {len(reads)}")
if len(reads) == 0:
    print("No reads found! Exiting."); sys.exit(1)

# ---- 3. EMBED READS ----
print("Embedding reads with DNABERT-6...")
model = FoundationModel("zhihan1996/DNA_bert_6")
read_embeds = []
BATCH = 16
for i in tqdm(range(0, len(reads), BATCH), desc="Read batches"):
    batch = reads[i:i+BATCH]
    read_embeds.append(model.encode(batch))
read_embeds = np.vstack(read_embeds)

# ---- 4. NEAREST NEIGHBOR SEARCH (FORCED 3-CLASS) ----
print("Fitting k-NN on only HLA-A, CDKN2A, KRAS full-gene embeddings...")
nbrs3 = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gene_embeds3)
genepred_idx3 = nbrs3.kneighbors(read_embeds, return_distance=False).flatten()
predicted3 = gene_labels3[genepred_idx3]

# --- Nearest Neighbor among all genes ---
print("Fitting k-NN on all full-gene embeddings (20K class)...")
nbrs_all = NearestNeighbors(n_neighbors=1, metric="cosine").fit(all_embeds)
genepred_idx_all = nbrs_all.kneighbors(read_embeds, return_distance=False).flatten()
predicted_all = gene_labels[genepred_idx_all]

# ---- 5. EVALUATE ----
all_labels3 = list(GENES.keys())
print("\n=== 3-class forced (HLA-A, KRAS, CDKN2A) classification ===")
print(classification_report(labels, predicted3, labels=all_labels3, target_names=all_labels3))
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
print("Confusion matrix:")
print(confusion_matrix(labels, predicted3, labels=all_labels3))

# ---- 6. UNIQUE READS DIAGNOSTIC ----
unique_reads = len(set(reads))
print(f"Number of unique input reads: {unique_reads} / {len(reads)}")
if unique_reads < len(reads) * 0.2:
    print("WARNING: Most reads seem to be duplicates (likely not informative for benchmarking).")

# ---- 7. WRITE ALL PREDICTIONS TO FILE ----
import csv
csv_path = "data/gene_nn_read_classification.csv"
with open(csv_path, "w", newline='') as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["read_idx", "true_gene", "predicted_gene_3class", "predicted_gene_20K", "cosine_distance_3class", "cosine_distance_20K", "sequence"])
    for i, (true_gene, seq, pred3, nnidx3, pred_all, nnidx_all) in enumerate(zip(labels, reads, predicted3, genepred_idx3, predicted_all, genepred_idx_all)):
        dist3 = nbrs3.kneighbors([read_embeds[i]], return_distance=True)[0][0][0]
        dist_all = nbrs_all.kneighbors([read_embeds[i]], return_distance=True)[0][0][0]
        writer.writerow([i+1, true_gene, pred3, pred_all, f"{dist3:.6f}", f"{dist_all:.6f}", seq])
print(f"\nPredictions written to {csv_path}")
