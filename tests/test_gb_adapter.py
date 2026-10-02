"""Actual RNABert architecture tests with tiny random local weights, no downloads."""
import numpy as np
import pytest
import torch
import csv
import json
from llm_mapper.embedders import EmbeddingConfig, create_embedder


def test_local_rnabert_mixed_lengths(tmp_path):
    pytest.importorskip("modelgenerator")
    from modelgenerator.huggingface_models.rnabert.configuration_rnabert import RNABertConfig
    from modelgenerator.huggingface_models.rnabert.modeling_rnabert import RNABertModel
    from modelgenerator.huggingface_models.rnabert.tokenization_rnabert import RNABertTokenizer
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]", "A", "C", "G", "T", "N", "U"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = RNABertTokenizer(str(tmp_path / "vocab.txt"))
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(42)
    config = RNABertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=32,
                           max_position_embeddings=512, hidden_dropout_prob=0.,
                           attention_probs_dropout_prob=0.)
    RNABertModel(config).save_pretrained(tmp_path)
    embedder = create_embedder(EmbeddingConfig(model=str(tmp_path)), "cpu")
    sequences = ["A" * 100, "C" * 150, "G" * 250]
    assert [len(embedder.tokenizer(s)["input_ids"]) for s in sequences] == [102, 152, 252]
    together = embedder.encode(sequences, batch_size=3)
    separate = np.concatenate([embedder.encode([s]) for s in sequences])
    assert together.shape == (3, 16)
    np.testing.assert_allclose(together, separate, atol=1e-6)
    np.testing.assert_array_equal(together, embedder.encode(sequences, batch_size=3))
    assert not np.allclose(together[0], together[1])
    sdpa = create_embedder(EmbeddingConfig(model=str(tmp_path), attention="sdpa"), "cpu")
    np.testing.assert_allclose(together, sdpa.encode(sequences, batch_size=3), atol=1e-6)
    from llm_mapper.__main__ import main
    fasta = tmp_path / "genes.fa"
    fasta.write_text(">geneA\n" + sequences[0] + "\n>geneC\n" + sequences[1] + "\n")
    reads = tmp_path / "reads.fq"
    reads.write_text("@read\n" + sequences[0] + "\n+\n" + "I" * 100 + "\n")
    index = tmp_path / "index"
    main(["index", str(fasta), str(index), "--model", str(tmp_path), "--device", "cpu"])
    predictions = tmp_path / "predictions.csv"
    main(["map", str(reads), str(index), str(predictions), "--device", "cpu"])
    with predictions.open() as handle:
        assert next(csv.DictReader(handle))["predicted_label"] == "geneA"
    report = tmp_path / "benchmark.json"
    main(["benchmark", str(reads), str(index), str(report), "--device", "cpu"])
    assert json.loads(report.read_text())["chunks"] == 1
