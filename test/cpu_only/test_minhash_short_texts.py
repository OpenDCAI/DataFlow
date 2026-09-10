import pandas as pd
import pytest

from dataflow.operators.general_text import MinHashDeduplicateFilter
from dataflow.utils.storage import DataFlowStorage


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
        return "memory://minhash-short-texts"


@pytest.mark.cpu
@pytest.mark.parametrize(
    "texts, ngram",
    [
        (["cat", "dog", "cat"], 5),
        (["北京", "上海", "北京"], 5),
        (["apples", "oranges", "apples"], 8),
    ],
)
def test_preserves_distinct_texts_shorter_than_ngram(texts, ngram):
    storage = MemoryStorage(texts)

    result_keys = MinHashDeduplicateFilter(ngram=ngram).run(
        storage, input_key="text"
    )

    assert storage.result["text"].tolist() == texts[:2]
    assert storage.result.index.tolist() == [0, 1]
    assert result_keys == ["minhash_deduplicated_label"]
    assert storage.result[result_keys[0]].tolist() == [1, 1]


@pytest.mark.cpu
def test_empty_texts_are_deduplicated_separately_from_short_texts():
    storage = MemoryStorage(["", "cat", "", "dog", "cat", ""])

    MinHashDeduplicateFilter().run(storage, input_key="text")

    assert storage.result["text"].tolist() == ["", "cat", "dog"]
    assert storage.result.index.tolist() == [0, 1, 3]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "texts",
    [
        ["aaaaa", "zzzzz", "aaaaa"],
        ["aaaaaaaaaa", "zzzzzzzzzz", "aaaaaaaaaa"],
    ],
)
def test_texts_at_or_above_ngram_keep_existing_deduplication(texts):
    storage = MemoryStorage(texts)

    MinHashDeduplicateFilter().run(storage, input_key="text")

    assert storage.result["text"].tolist() == texts[:2]


@pytest.mark.cpu
def test_character_mode_keeps_existing_deduplication():
    storage = MemoryStorage(["cat", "dog", "cat"])

    MinHashDeduplicateFilter(use_n_gram=False).run(storage, input_key="text")

    assert storage.result["text"].tolist() == ["cat", "dog"]


@pytest.mark.cpu
def test_reusing_operator_does_not_share_deduplication_state():
    operator = MinHashDeduplicateFilter()

    for _ in range(2):
        storage = MemoryStorage(["cat", "dog", "cat"])
        operator.run(storage, input_key="text")
        assert storage.result["text"].tolist() == ["cat", "dog"]
