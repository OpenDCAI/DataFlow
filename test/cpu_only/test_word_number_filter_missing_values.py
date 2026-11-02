import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from dataflow.operators.general_text.filter.word_number_filter import WordNumberFilter
from dataflow.utils.storage import LazyFileStorage


def _storage_with_texts(texts):
    root = tempfile.mkdtemp()
    source = os.path.join(root, "input.jsonl")
    pd.DataFrame({"text": texts}).to_json(
        source, orient="records", lines=True, force_ascii=False
    )
    storage = LazyFileStorage(
        first_entry_file_name=source,
        cache_path=os.path.join(root, "cache"),
        file_name_prefix="cache",
        cache_type="jsonl",
        save_on_exit=False,
    )
    storage.step()
    return storage


@pytest.mark.cpu
def test_missing_cells_are_counted_as_zero_words():
    # A null cell reaches the operator whenever an upstream generator returns
    # nothing for a row, or a left join does not match. It must not abort the run.
    storage = _storage_with_texts(["one two three", None, np.nan, "four five"])

    WordNumberFilter(min_words=0, max_words=100).run(storage, input_key="text")

    written = storage.step().read()
    assert written["word_number_filter_label"].tolist() == [3, 0, 0, 2]
    assert written["text"].iloc[0] == "one two three"
    assert pd.isna(written["text"].iloc[1])
    assert pd.isna(written["text"].iloc[2])
    assert written["text"].iloc[3] == "four five"


@pytest.mark.cpu
def test_rows_outside_the_range_are_dropped_in_original_order():
    storage = _storage_with_texts(["one two three", None, "four five"])

    WordNumberFilter(min_words=2, max_words=10).run(storage, input_key="text")

    written = storage.step().read()
    assert written["text"].tolist() == ["one two three", "four five"]
    assert written["word_number_filter_label"].tolist() == [3, 2]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "cell, expected",
    [
        (None, 0),
        (np.nan, 0),
        (pd.NA, 0),
        (42, 0),
        ("", 0),
        ("   ", 0),
        ("one", 1),
        ("one  two\tthree", 3),
    ],
)
def test_count_words_treats_non_string_cells_as_empty(cell, expected):
    assert WordNumberFilter._count_words(cell) == expected
