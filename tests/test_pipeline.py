import gzip
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from llm_mapper.sequences import SequenceReader, SequenceRecord, ReadChunker, batches
from llm_mapper.embedders import EmbeddingConfig, SequenceEmbedder
from llm_mapper.embedders.transformer import TransformerEmbedder, pool_hidden
from llm_mapper.devices import resolve_device
from llm_mapper.index import build_index, VectorIndex, map_records


class CountEmbedder(SequenceEmbedder):
    """Deterministic test double only; not a biological model or benchmark."""
    config = EmbeddingConfig(model="test-counts", revision="test-v1")
    device = "cpu"

    def encode(self, sequences, batch_size=8):
        return np.array([[s.count(c) for c in "ACGT"] for s in sequences], dtype=np.float32)


@pytest.mark.parametrize("length", [100, 150, 250])
@pytest.mark.parametrize("format", ["fasta", "fastq", "bam"])
def test_read_lengths(tmp_path, length, format):
    sequence = ("ACGT" * 100)[:length]
    path = tmp_path / f"reads.{format}"
    if format == "fasta":
        path.write_text(f">read description\n{sequence[:50]}\n{sequence[50:]}\n")
    elif format == "fastq":
        path.write_text(f"@read description\n{sequence[:50]}\n{sequence[50:]}\n+\n{'@' * length}\n")
    else:
        import pysam
        with pysam.AlignmentFile(path, "wb", header={"HD": {"VN": "1.6"}}) as bam:
            read = pysam.AlignedSegment()
            read.query_name, read.query_sequence, read.flag = "read", sequence, 4
            bam.write(read)
    records = list(SequenceReader(path))
    assert len(records) == 1
    assert records[0].sequence == sequence
    assert list(ReadChunker(250)(records)) == records


def test_bam_orientation_and_filtering(tmp_path):
    import pysam
    path = tmp_path / "reads.bam"
    with pysam.AlignmentFile(path, "wb", header={"SQ": [{"SN": "chr1", "LN": 1000}]}) as bam:
        for flag in [16 | 64, 256, 2048]:
            read = pysam.AlignedSegment()
            read.query_name, read.query_sequence, read.flag = "r", "ACCC", flag
            bam.write(read)
    records = list(SequenceReader(path))
    assert [(r.id, r.sequence) for r in records] == [("r/1", "GGGT")]


def test_gzip_csv_and_invalid_fastq(tmp_path):
    path = tmp_path / "reads.fq.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("@r\nACGT\n+\n!!\n!!\n")
    assert next(iter(SequenceReader(path))).sequence == "ACGT"
    malformed = tmp_path / "bad.fq"
    for text in ["@r\nACGT\n+\n!!!\n", "@r\nACGT\n+\n!!!!!\n", "ACGT\n"]:
        malformed.write_text(text)
        with pytest.raises(ValueError):
            list(SequenceReader(malformed))
    csv = tmp_path / "genes.csv"
    csv.write_text("gene_symbol,sequence\nGENE,acgt\n")
    assert next(iter(SequenceReader(csv))).label == "GENE"


def test_chunk_coverage_and_lazy_batches():
    record = SequenceRecord("r", "A" * 601, "gene")
    chunks = list(ReadChunker(250)([record]))
    assert [(c.start, c.end) for c in chunks] == [(0, 250), (250, 500), (500, 601)]
    assert "".join(c.sequence for c in chunks) == record.sequence
    overlap = list(ReadChunker(250, 100)([record]))
    assert overlap[-1].end == 601
    assert all(c.label == "gene" for c in overlap)
    consumed = []
    def source():
        for i in range(100):
            consumed.append(i)
            yield i
    assert next(batches(source(), 3)) == [0, 1, 2]
    assert consumed == [0, 1, 2]
    with pytest.raises(ValueError):
        list(batches([], 0))
    with pytest.raises(ValueError):
        ReadChunker(250, 251)


def test_masked_pooling_excludes_special_and_padding():
    hidden = torch.tensor([[[99., 99.], [1., 3.], [3., 5.], [88., 88.], [77., 77.]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    torch.testing.assert_close(pool_hidden(hidden, attention, special, "mean"), torch.tensor([[2., 4.]]))


def test_transformer_batching_and_no_truncation():
    calls = []
    class Tokenizer:
        def __call__(self, seqs, **kwargs):
            assert kwargs["truncation"] is False
            calls.append(len(seqs))
            width = max(map(len, seqs)) + 2
            ids = torch.zeros((len(seqs), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            special = torch.ones_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, 1:len(seq)+1] = 2
                mask[i, :len(seq)+2] = 1
                special[i, 1:len(seq)+1] = 0
            return dict(input_ids=ids, attention_mask=mask, special_tokens_mask=special)
    class Model:
        def __call__(self, input_ids, attention_mask):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))
    model = object.__new__(TransformerEmbedder)
    model.config, model.device = EmbeddingConfig(), "cpu"
    model.tokenizer, model.model = Tokenizer(), Model()
    model.dimension, model.context_length = 1, 4000
    result = model.encode(["A" * n for n in [100, 150, 250]], batch_size=2)
    np.testing.assert_equal(result, np.full((3, 1), 2))
    assert calls == [2, 1]
    assert model.encode([]).shape == (0, 1)
    with pytest.raises(ValueError, match="exceeds"):
        model.encode(["A" * 4000])


def test_device_detection():
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device() == "mps"
        assert resolve_device("cpu") == "cpu"
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("metal")


def test_index_roundtrip_lengths_and_mismatch(tmp_path):
    embedder = CountEmbedder()
    references = [SequenceRecord("geneA", "A" * 600), SequenceRecord("geneC", "C" * 600)]
    path = tmp_path / "index"
    manifest = build_index(iter(references), embedder, path, ReadChunker(250), batch_size=2)
    assert manifest["count"] == 6
    index = VectorIndex(path, embedder)
    reads = [SequenceRecord(str(n), "C" * n) for n in [100, 150, 250]]
    predictions = list(map_records(iter(reads), index, batch_size=2))
    assert [p["read_end"] for p in predictions] == [100, 150, 250]
    assert all(p["predicted_label"] == "geneC" and p["cosine_distance"] == 0 for p in predictions)
    other = CountEmbedder()
    other.config = replace(other.config, pooling="cls")
    with pytest.raises(ValueError, match="mismatch"):
        VectorIndex(path, other)
    with pytest.raises(FileExistsError):
        build_index(references, embedder, path)


def test_index_failure_does_not_publish(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        build_index([], CountEmbedder(), tmp_path / "empty")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="Zero"):
        build_index([SequenceRecord("n", "NNN")], CountEmbedder(), tmp_path / "zero")
    assert list(tmp_path.iterdir()) == []


def test_evo2_requires_validated_cpu_loader():
    from llm_mapper.embedders.evo2 import Evo2Embedder
    with pytest.raises(RuntimeError, match="validated"):
        Evo2Embedder(EmbeddingConfig(backend="evo2", model="evo2_7b", layer="blocks.28.mlp.l3"), "cpu")


def test_evo2_loader_boundary_and_token_lengths():
    import sys
    from types import ModuleType
    from llm_mapper.embedders.evo2 import Evo2Embedder
    lengths = []
    class Wrapper:
        model = torch.nn.Identity()
        tokenizer = SimpleNamespace(tokenize=lambda sequence: [ord(c) for c in sequence])
        def __call__(self, tokens, return_embeddings, layer_names):
            lengths.append(tokens.shape[1])
            return None, {layer_names[0]: tokens.float().unsqueeze(-1)}
    module = ModuleType("test_evo_loader")
    module.load = lambda config, device: Wrapper()
    config = EmbeddingConfig(backend="evo2", model="fixture", layer="test",
                             loader="test_evo_loader:load")
    original_cuda = torch.cuda
    with patch.dict(sys.modules, {"test_evo_loader": module}):
        model = Evo2Embedder(config, "cpu")
        result = model.encode(["A" * n for n in [100, 150, 250]])
    assert lengths == [100, 150, 250]
    np.testing.assert_array_equal(result, np.full((3, 1), ord("A")))
    assert torch.cuda is original_cuda


def test_benchmark_truth_and_memory(tmp_path):
    from llm_mapper.benchmark import benchmark
    embedder = CountEmbedder()
    build_index([SequenceRecord("geneA", "A" * 250)], embedder, tmp_path / "index")
    path = tmp_path / "reads.fa"
    path.write_text(">r\n" + "A" * 150 + "\n")
    result = benchmark(path, VectorIndex(tmp_path / "index", embedder), truth={"r": "geneA"})
    assert result["accuracy_per_chunk"] == 1
    assert result["chunks"] == 1 and result["seconds"] > 0
    assert result["sampled_peak_rss_bytes"] > 0
    with pytest.raises(KeyError):
        benchmark(path, VectorIndex(tmp_path / "index", embedder), truth={})
