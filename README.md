# LLM-mapper: Sequence Embedding and RNA-seq Gene Mapping Pipeline

This pipeline enables LLM-based gene embedding, RAG search, and evaluation for RNA/DNA sequencing using models such as DNABERT, AIDO (via modelgenerator), Evo2, and NucleotideTransformer.

## Setup

1. **Install environment (Conda + pip hybrid)**
    ```sh
    conda env create -f environment.yml
    conda activate llm-mapper
    pip install -r requirements.txt
    ```
2. **Prepare input data**
    - Place genome FASTA, GTF, test BAMs, and processed embedding batches in `data/` as described in scripts/.

## Usage

- **Embed all genes/CDS with any model** (set model in the script):
    ```python
    # DNABERT
    model = FoundationModel("zhihan1996/DNA_bert_6", use_kmer=True)
    # AIDO via ModelGenerator
    embedder = get_embedder("aido_rna_650m_cds", use_kmer=False)
    ```
- **Batch embed genes (creates merged embedding arrays):**
    ```sh
    python llm_mapper/embed_kmers.py data/genome/gene_sequences.csv data/genome/full_gene_metadata.csv
    python llm_mapper/merge_embedding_batches_memmap.py
    ```
- **UMAP visualization:**
    ```sh
    python llm_mapper/plot_umap_genes.py
    ```
- **Nearest neighbor gene classification of reads:**
    ```sh
    python llm_mapper/eval_reads_gene_nn.py
    ```

## Model Switching & Customization
- To swap LLM backbone, set `MODEL_NAME` and `use_kmer` in each key script.
- See comments in scripts/ for toggling AIDO or DNABERT pipelines.

## Repo Layout

```
LLM_mapper/
├── llm_mapper/               # Python package (core, CLI, helpers)
├── scripts/                  # Utility pipelines
├── tests/                    # (Optional) unit tests
├── data/                     # (User, not committed)
├── environment.yml           # Conda+pip hybrid
├── requirements.txt
├── README.md
└── .gitignore
```

---

- For more models/info: [Huggingface Model Hub](https://huggingface.co/models)
- See scripts/ and llm_mapper/ for full workflow stages and options
