import math
import sys

import numpy as np
import pandas as pd
import pytest

from dataflow import get_logger
from dataflow.operators.general_text import SemDeduplicateFilter
from dataflow.utils.storage import DataFlowStorage

sem_dedup = sys.modules[SemDeduplicateFilter.__module__]


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://sem-dedup"


def _unit(degrees):
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def _operator(eps=0.05, batch_size=256, block_size=1024):
    """Build the operator without touching the network: __init__ downloads a BERT model."""
    operator = SemDeduplicateFilter.__new__(SemDeduplicateFilter)
    operator.logger = get_logger()
    operator.eps = eps
    operator.batch_size = batch_size
    operator.block_size = block_size
    operator.device = "cpu"
    operator.model = None
    operator.tokenizer = None
    operator.model_name = "stub"
    operator.model_cache_dir = "stub"
    return operator


def _stub_embeddings(monkeypatch, vectors, calls=None):
    def fake(texts, tokenizer, model, device, batch_size=256):
        if calls is not None:
            calls.append((len(texts), batch_size))
        return np.asarray(vectors, dtype="float32")

    monkeypatch.setattr(sem_dedup, "get_text_embedding", fake)


@pytest.mark.cpu
def test_a_chain_of_near_duplicates_keeps_the_tail(monkeypatch):
    # alpha ~ beta (10 deg) and beta ~ gamma (10 deg), but alpha !~ gamma (20 deg).
    # Only beta is a duplicate, so dropping it must not take gamma with it.
    _stub_embeddings(monkeypatch, [_unit(0), _unit(10), _unit(20)])
    storage = MemoryStorage(pd.DataFrame({"text": ["alpha", "beta", "gamma"]}))

    _operator().run(storage, input_key="text")

    assert storage.result["text"].tolist() == ["alpha", "gamma"]
    assert storage.result["minhash_deduplicated_label"].tolist() == [1, 1]


@pytest.mark.cpu
def test_a_chain_is_not_collapsed_into_a_single_row(monkeypatch):
    # Every neighbour is 10 deg apart and the threshold is 0.95, so the chain
    # 0-10-20-...-60 deg keeps one row every two steps instead of only the first.
    _stub_embeddings(monkeypatch, [_unit(degrees) for degrees in range(0, 70, 10)])
    storage = MemoryStorage(pd.DataFrame({"text": [f"t{i}" for i in range(7)]}))

    _operator().run(storage, input_key="text")

    assert storage.result["text"].tolist() == ["t0", "t2", "t4", "t6"]


@pytest.mark.cpu
@pytest.mark.parametrize("block_size", [1, 3, 64, 1024])
def test_blocked_scoring_matches_a_row_by_row_reference(block_size):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 8)).astype("float32")
    embeddings[40] = embeddings[7] + 0.01
    embeddings[41] = embeddings[40] + 0.01

    expected = _row_by_row_labels(embeddings, threshold=0.9)

    assert np.array_equal(
        sem_dedup.find_duplicate_labels(embeddings, 0.9, block_size=block_size), expected
    )


def _row_by_row_labels(embeddings, threshold):
    unit = sem_dedup._unit_rows(embeddings)
    labels = np.ones(unit.shape[0], dtype=int)
    kept = []
    for index in range(unit.shape[0]):
        if kept and np.max(unit[kept] @ unit[index]) >= threshold:
            labels[index] = 0
        else:
            kept.append(index)
    return labels


@pytest.mark.cpu
def test_eps_is_the_margin_below_one(monkeypatch):
    # Cosine similarity between the two rows is cos(15 deg) ~= 0.966.
    vectors = [_unit(0), _unit(15)]

    _stub_embeddings(monkeypatch, vectors)
    loose = MemoryStorage(pd.DataFrame({"text": ["a", "b"]}))
    _operator(eps=0.05).run(loose, input_key="text")
    assert loose.result["text"].tolist() == ["a"]

    _stub_embeddings(monkeypatch, vectors)
    strict = MemoryStorage(pd.DataFrame({"text": ["a", "b"]}))
    _operator(eps=0.02).run(strict, input_key="text")
    assert strict.result["text"].tolist() == ["a", "b"]


@pytest.mark.cpu
def test_texts_are_embedded_in_batches(monkeypatch):
    calls = []
    _stub_embeddings(monkeypatch, [_unit(0), _unit(90), _unit(180)], calls=calls)
    storage = MemoryStorage(pd.DataFrame({"text": ["a", "b", "c"]}))

    _operator(batch_size=2).run(storage, input_key="text")

    assert calls == [(3, 2)]


@pytest.mark.cpu
def test_embedding_helper_splits_the_input():
    import torch

    seen = []

    class StubModel:
        def __call__(self, input_ids, **kwargs):
            seen.append(len(input_ids))
            return type("Out", (), {"last_hidden_state": torch.zeros(len(input_ids), 1, 4)})()

    class StubInputs(dict):
        def to(self, device):
            return self

    def fake_tokenizer(texts, **kwargs):
        return StubInputs(input_ids=texts)

    embeddings = sem_dedup.get_text_embedding(
        ["t"] * 5, fake_tokenizer, StubModel(), "cpu", batch_size=2
    )

    assert seen == [2, 2, 1]
    assert embeddings.shape == (5, 4)


@pytest.mark.cpu
def test_an_empty_dataframe_is_written_back_untouched(monkeypatch):
    calls = []
    _stub_embeddings(monkeypatch, [], calls=calls)
    storage = MemoryStorage(pd.DataFrame({"text": pd.Series([], dtype=object)}))

    result = _operator().run(storage, input_key="text")

    assert calls == []
    assert result == ["minhash_deduplicated_label"]
    assert storage.result.empty
