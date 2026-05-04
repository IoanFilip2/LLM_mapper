import torch
from typing import List, Union
from transformers import AutoModel, AutoTokenizer, AutoConfig
# Register swiglu for AIDO RNA model before config/model instantiation
try:
    from transformers.activations import ACT2FN
    class Swiglu(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            x1, x2 = x.chunk(2, dim=-1)
            return x1 * torch.sigmoid(x2)
    ACT2FN["swiglu"] = Swiglu()
except Exception as e:
    print("Could not register swiglu:", e)

class FoundationModel:
    def __init__(self, model_name: str, device: str = None, use_kmer: bool = True):
        self.model_name = model_name
        self.use_kmer = use_kmer
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if model_name == "genbio-ai/AIDO.RNA-650M-CDS":
            from transformers import BertTokenizer, BertModel
            print("Using BertTokenizer for AIDO.RNA-650M-CDS")
            self.tokenizer = BertTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # No AutoConfig, just load with BertModel/trust_remote_code
            self.model = BertModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
                config.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
            self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True).to(self.device)

    def kmerize(self, seq: str, k: int = 6):
        """
        Convert a DNA sequence string to a k-mer (overlapping, separated by space) string.
        """
        seq = seq.upper().replace(' ', '').replace('\n', '')
        return ' '.join(seq[i:i+k] for i in range(len(seq)-k+1))

    def encode(self, sequences: Union[str, List[str]], batch_size: int = 8, k: int = 6, verbose: bool = False):
        if isinstance(sequences, str):
            sequences = [sequences]
        if self.use_kmer:
            seqs_for_input = [self.kmerize(seq, k) for seq in sequences]
        else:
            seqs_for_input = [seq.upper() for seq in sequences]
        if verbose:
            print("Original example seq:", sequences[0][:80] + ("..." if len(sequences[0]) > 80 else ""))
            if self.use_kmer:
                print("K-merized:", seqs_for_input[0][:80] + ("..." if len(seqs_for_input[0]) > 80 else ""))
            sample_tok = self.tokenizer(
                seqs_for_input[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            print("Tokenized IDs:", sample_tok['input_ids'])
            if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                print("Tokens:", self.tokenizer.convert_ids_to_tokens(sample_tok['input_ids'][0]))
        encodings = self.tokenizer(
            seqs_for_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(
                **encodings,
                output_hidden_states=False,
                output_attentions=False
            )
            # For DNABERT, the [CLS] is at position 0
            # shape: (batch, seq_len, hidden)
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, (tuple, list)):
                emb = outputs[0][:, 0, :]
            else:
                raise RuntimeError("Model output does not have 'last_hidden_state'.")
        return emb.detach().cpu().numpy()

if __name__ == "__main__":
    import sys
    model = FoundationModel(model_name=sys.argv[1] if len(sys.argv) > 1 else "Salesforce/evo-2-p-t512-nt")
    print(f"Device: {model.device}")
    testseq = "ACGTACGTACGTACGTACGTACGTACGTACGT"
    embedding = model.encode([testseq, testseq[::-1]])
    print(f"Embedding shape: {embedding.shape}")
