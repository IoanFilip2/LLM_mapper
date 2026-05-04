from transformers import AutoTokenizer

def kmerize(seq, k=6):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    return ' '.join([seq[i:i+k] for i in range(len(seq) - k + 1)])

tok = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)

seq1 = "AACGTGGCCTGAAGTAATTAGGTTG"
seq2 = "TTTTTTTTTTTTTTTTTTTTTTTTTT"
kmer_seq1 = kmerize(seq1)
kmer_seq2 = kmerize(seq2)

print(f"K-merized 1: {kmer_seq1}")
print(f"K-merized 2: {kmer_seq2}")
tok_out1 = tok(kmer_seq1, add_special_tokens=True)
tok_out2 = tok(kmer_seq2, add_special_tokens=True)

print("Tokenizer Output IDs 1:", tok_out1['input_ids'])
print("Tokenizer Output IDs 2:", tok_out2['input_ids'])
if hasattr(tok, 'convert_ids_to_tokens'):
    print("Tokenizer Tokens 1:", tok.convert_ids_to_tokens(tok_out1['input_ids']))
    print("Tokenizer Tokens 2:", tok.convert_ids_to_tokens(tok_out2['input_ids']))
# Check for [UNK] tokens
print(f"Any [UNK] in seq1 tokens?", '[UNK]' in tok.convert_ids_to_tokens(tok_out1['input_ids']))
print(f"Any [UNK] in seq2 tokens?", '[UNK]' in tok.convert_ids_to_tokens(tok_out2['input_ids']))
