"""Padding must not change embeddings or semantic deduplication decisions."""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from dataflow.operators.general_text import SemDeduplicateFilter
from dataflow.operators.general_text.filter.sem_deduplicate_filter import (
    get_text_embedding,
)
from dataflow.utils.storage import DataFlowStorage

pytestmark = pytest.mark.cpu


@pytest.fixture(scope="module")
def operator(tmp_path_factory):
    # A real local BERT keeps these tests independent of downloaded models.
    model_dir = tmp_path_factory.mktemp("semantic-dedup-bert")
    vocab = model_dir / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncat\ndog\nthe\n", encoding="utf-8"
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_dir, model_max_length=64, local_files_only=True
    )
    tokenizer.save_pretrained(model_dir)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BertModel(
            BertConfig(
                vocab_size=8,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        )
    model.save_pretrained(model_dir)
    return SemDeduplicateFilter(model_name=str(model_dir), device="cpu", eps=0.02)


class MemoryStorage(DataFlowStorage):
    def __init__(self, texts):
        self.dataframe = pd.DataFrame({"text": texts})
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://semantic-dedup"


@pytest.mark.parametrize("text", ["cat", "", "cat dog"])
def test_embedding_is_independent_of_other_text_lengths(operator, text):
    solo = get_text_embedding([text], operator.tokenizer, operator.model, "cpu")
    batched = get_text_embedding(
        [text, "the " * 30], operator.tokenizer, operator.model, "cpu"
    )
    np.testing.assert_allclose(batched[:1], solo, atol=1e-6, rtol=1e-6)
    assert np.isfinite(batched).all()


def test_unpadded_embeddings_keep_existing_mean_pooling(operator):
    texts = ["cat", "dog"]
    inputs = operator.tokenizer(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        expected = operator.model(**inputs).last_hidden_state.mean(dim=1).numpy()
    actual = get_text_embedding(texts, operator.tokenizer, operator.model, "cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "input_args", [{"input_key": "text"}, {"input_keys": ["text"]}]
)
def test_long_row_does_not_turn_short_texts_into_duplicates(operator, input_args):
    short = MemoryStorage(["cat", "dog", "cat"])
    padded = MemoryStorage(["cat", "dog", "cat", "the " * 30])
    assert operator.run(short, output_key="unique", **input_args) == ["unique"]
    assert operator.run(padded, output_key="unique", **input_args) == ["unique"]
    assert short.result.index.tolist() == [0, 1]
    assert padded.result.index.tolist() == [0, 1, 3]
    assert padded.result["unique"].tolist() == [1, 1, 1]
