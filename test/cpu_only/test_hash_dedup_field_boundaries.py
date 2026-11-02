"""Exact deduplication must preserve the boundaries of selected text fields."""

import pandas as pd
import pytest

from dataflow.operators.general_text import HashDeduplicateFilter
from dataflow.utils.storage import DataFlowStorage


pytestmark = pytest.mark.cpu


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
        return "memory://hash-dedup-field-boundaries"


@pytest.mark.parametrize("hash_func", ["md5", "sha256", "xxh3"])
@pytest.mark.parametrize("answer_key", ["answer", "", '答案"\n:'])
def test_distinct_fields_survive_embedded_field_markers(hash_func, answer_key):
    # Both rows produce the same text when fields are joined with a field-name
    # delimiter. The third row is a real duplicate and must still be removed.
    marker = f"\n{answer_key}:\n"
    dataframe = pd.DataFrame(
        {
            "question": [f"x{marker}y", "x", f"x{marker}y"],
            answer_key: ["z", f"y{marker}z", "z"],
            "id": ["first", "second", "duplicate"],
        },
        index=[7, 2, 9],
    )
    storage = MemoryStorage(dataframe)

    result_keys = HashDeduplicateFilter(hash_func=hash_func).run(
        storage, input_keys=["question", answer_key], output_key="unique"
    )

    assert storage.result["id"].tolist() == ["first", "second"]
    assert storage.result.index.tolist() == [7, 2]
    assert storage.result["unique"].tolist() == [1, 1]
    assert result_keys == ["unique"]


@pytest.mark.parametrize("hash_func", ["md5", "sha256", "xxh3"])
def test_empty_unicode_and_escaped_fields_remain_distinct(hash_func):
    pairs = [
        ("", ""),
        ("", "a"),
        ("a", ""),
        ("\x00", ""),
        ("", "\x00"),
        ("中文😀", '"\\\n'),
        ('"\\\n', "中文😀"),
        ("é", "e\u0301"),
        ("e\u0301", "é"),
    ]
    storage = MemoryStorage(pd.DataFrame(pairs + pairs, columns=["left", "right"]))

    HashDeduplicateFilter(hash_func=hash_func).run(
        storage, input_keys=["left", "right"]
    )

    assert list(zip(storage.result["left"], storage.result["right"])) == pairs
    assert storage.result.index.tolist() == list(range(len(pairs)))


@pytest.mark.parametrize("hash_func", ["md5", "sha256", "xxh3"])
@pytest.mark.parametrize("value", [1, None, True, float("nan"), ["text"]])
def test_multi_field_values_keep_existing_string_conversion(hash_func, value):
    storage = MemoryStorage(
        pd.DataFrame(
            {"left": [value, str(value)], "right": ["same", "same"]},
            dtype=object,
        )
    )

    HashDeduplicateFilter(hash_func=hash_func).run(
        storage, input_keys=["left", "right"]
    )

    # Multi-field inputs have always compared each value's string form.
    assert storage.result.index.tolist() == [0]


@pytest.mark.parametrize("hash_func", ["md5", "sha256", "xxh3"])
@pytest.mark.parametrize(
    "key_argument", [{"input_key": "text"}, {"input_keys": ["text"]}]
)
def test_single_field_inputs_keep_existing_behavior(hash_func, key_argument):
    texts = ["hello", "hello\nworld", "hello", "中文😀", "", ""]
    storage = MemoryStorage(pd.DataFrame({"text": texts}))

    HashDeduplicateFilter(hash_func=hash_func).run(storage, **key_argument)

    assert storage.result["text"].tolist() == ["hello", "hello\nworld", "中文😀", ""]
    assert storage.result.index.tolist() == [0, 1, 3, 4]
