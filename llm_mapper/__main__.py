import typer
from llm_mapper.foundation import FoundationModel

def main(
    model_name: str = typer.Option("Salesforce/evo-2-p-t512-nt", help="HF model for nucleotide embedding"),
    seq: str = typer.Argument(..., help="DNA/RNA sequence to embed")
):
    fm = FoundationModel(model_name=model_name)
    print(f"Device: {fm.device}")
    emb = fm.encode([seq])
    print(f"Embedding (shape={emb.shape}):\n{emb}")

if __name__ == "__main__":
    typer.run(main)
