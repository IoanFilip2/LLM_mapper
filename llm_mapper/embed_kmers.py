import csv
import sys
import os
csv.field_size_limit(sys.maxsize)
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))
def get_embedder(model_name, use_kmer=True):
    if model_name.lower() in ["aido_rna_650m_cds", "genbio-ai/AIDO.RNA-650M-CDS"]:
        from modelgenerator.tasks import Embed
        embedder = Embed.from_config({"model.backbone": "aido_rna_650m_cds"}).eval()
        def embedding_func(seqs):
            batch = embedder.transform({"sequences": seqs})
            return embedder(batch).detach().cpu().numpy()
        return embedding_func
    else:
        from foundation import FoundationModel
        model = FoundationModel(model_name, use_kmer=use_kmer)
        def embedding_func(seqs):
            return model.encode(seqs)
        return embedding_func

K_LIST = [60, 100, 500]
SAMPLES_PER_K = 33
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EMBED_DIM = 768  # For DNA_bert_6
BATCH_SIZE = 10
BATCH_DIR = "data/genome/gene_kmer_embedding_batches"
os.makedirs(BATCH_DIR, exist_ok=True)

def sample_subsequences(seq, k, n):
    # Uniformly sample n subsequences of length k from seq; spread across sequence.
    if len(seq) < k: return []
    if n == 1 or len(seq)-k < 1:
        idxs = [0]
    else:
        step = (len(seq)-k)/(n-1)
        idxs = [int(round(i*step)) for i in range(n)]
    return [(s, s+k, seq[s:s+k]) for s in idxs]

def append_metadata(meta_file, rows):
    writeheader = not Path(meta_file).exists()
    with open(meta_file, 'a', newline='') as outf:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(outf, fieldnames=fields)
        if writeheader:
            writer.writeheader()
        writer.writerows(rows)

def main():
    gene_csv = sys.argv[1] if len(sys.argv) > 1 else "data/genome/gene_sequences.csv"
    out_meta = sys.argv[2] if len(sys.argv) > 2 else "data/genome/full_gene_metadata.csv"
    out_batch_dir = "data/genome/full_gene_embedding_batches"
    os.makedirs(out_batch_dir, exist_ok=True)
    # For DNABERT-6: (uncomment as needed)
    # MODEL_NAME = "zhihan1996/DNA_bert_6"
    # embedder = get_embedder(MODEL_NAME, use_kmer=True)
    # For AIDO.RNA-650M-CDS (modelgenerator Embed):
    MODEL_NAME = "aido_rna_650m_cds"
    embedder = get_embedder(MODEL_NAME, use_kmer=False)
    batch_metadata = []
    batch_embeddings = []
    batch_idx = 0
    BATCH_SIZE = 500
    with open(gene_csv) as f:
        reader = list(csv.DictReader(f))
        for i, row in enumerate(tqdm(reader, desc='Full-gene', unit='gene')):
            # Only embed the full gene sequence, no k-mer subsampling
            seq = row['sequence'].upper()
            seq_len = len(seq)
            if seq_len >= 1:
                emb = embedder([seq])[0]
                batch_metadata.append({k: row[k] for k in row.keys()})
                batch_embeddings.append(emb)
            # Every BATCH_SIZE genes, save partial results:
            if (i+1) % BATCH_SIZE == 0 or (i+1) == len(reader):
                if batch_metadata:
                    append_metadata(out_meta, batch_metadata)
                np.save(f"{out_batch_dir}/embeddings_batch_{batch_idx}.npy", np.stack(batch_embeddings))
                batch_metadata.clear()
                batch_embeddings.clear()
                batch_idx += 1
                print(f"[BATCH {batch_idx}] Progress saved: gene {i+1}/{len(reader)}")
    print(f"DONE. All batch results saved in {out_batch_dir} and metadata in {out_meta}")
    print("To merge all embeddings: np.concatenate on all .npy files in batch dir.")


if __name__ == "__main__":
    main()
