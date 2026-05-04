import pandas as pd

csv_path = "data/genome/gene_sequences.csv"
df = pd.read_csv(csv_path)

n_total = len(df)
sequences = df['sequence']
n_unique = len(set(sequences))
n_all_n = sum(seq and set(seq) == {'N'} for seq in sequences)
n_empty = sum(seq == "" for seq in sequences)

print(f"Total genes: {n_total}")
print(f"Unique sequences: {n_unique}")
print(f"Empty sequences: {n_empty}")
print(f"All-N sequences: {n_all_n}")

# Show a sample of non-empty, non-all-N sequences (first 3)
real = [seq for seq in sequences if seq and not (set(seq) == {'N'})]
print("\nSample of extracted gene sequences (first 3 shown):")
for seq in real[:3]:
    print(seq[:80] + ("..." if len(seq) > 80 else ""))
