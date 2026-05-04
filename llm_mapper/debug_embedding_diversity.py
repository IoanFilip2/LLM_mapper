import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(__file__).replace('/debug_embedding_diversity.py', ''))

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

csv_path = "data/genome/gene_sequences.csv"
df = pd.read_csv(csv_path)
real_seqs = [seq for seq in df['sequence'] if seq and not (set(seq) == {'N'})][:5]
nonly_seq = "N" * 30

# Switch here for model
MODEL_NAME = "aido_rna_650m_cds"  # or "zhihan1996/DNA_bert_6"
embedder = get_embedder(MODEL_NAME, use_kmer=False)

print("Testing embedding diversity for model:", MODEL_NAME)
test_seqs = real_seqs + [nonly_seq]
for i, s in enumerate(test_seqs):
    print(f"Seq {i}: {s[:30]}{'...' if len(s) > 30 else ''}")

# Embed one sequence at a time to avoid OOM (especially for large models like AIDO)
embeddings = []
for seq in test_seqs:
    emb = embedder([seq])[0]
    embeddings.append(emb)
embeddings = np.vstack(embeddings)

print("\nPairwise L2 diffs among embeddings:")
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        diff = np.linalg.norm(embeddings[i] - embeddings[j])
        print(f"Seq {i} vs Seq {j}: ||diff|| = {diff:.6f}")