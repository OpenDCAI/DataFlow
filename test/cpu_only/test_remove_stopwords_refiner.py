import os
import tempfile

import numpy as np
import pandas as pd
import pytest

nltk = pytest.importorskip("nltk")

from nltk.corpus import stopwords

from dataflow.operators.general_text.refine.remove_stopwords_refiner import (
    RemoveStopwordsRefiner,
)
from dataflow.utils.storage import LazyFileStorage


def _require_corpus():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        pytest.skip("nltk stopwords corpus is not installed")


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
def test_stopwords_are_removed():
    _require_corpus()
    storage = _storage_with_texts(["The quick brown fox", "It is what it is"])

    RemoveStopwordsRefiner().run(storage, input_key="text")

    written = storage.step().read()
    assert written["text"].tolist() == ["quick brown fox", ""]


@pytest.mark.cpu
def test_missing_cells_are_preserved():
    # A null cell reaches the operator whenever an upstream generator returns
    # nothing for a row. It has no stopwords to remove, so it must pass through
    # untouched instead of aborting the step.
    _require_corpus()
    storage = _storage_with_texts(["The fox", None, np.nan, "A dog"])

    RemoveStopwordsRefiner().run(storage, input_key="text")

    written = storage.step().read()
    assert written["text"].iloc[0] == "fox"
    assert pd.isna(written["text"].iloc[1])
    assert pd.isna(written["text"].iloc[2])
    assert written["text"].iloc[3] == "dog"


@pytest.mark.cpu
def test_corpus_is_read_once_per_operator(monkeypatch):
    # Reading and re-parsing the corpus for every row costs ~0.7 ms per row,
    # which dominates the real work on large tables. The operator must load it
    # once at construction time and reuse it.
    _require_corpus()
    loads = []
    real_words = stopwords.words

    def counting_words(language):
        loads.append(language)
        return real_words(language)

    monkeypatch.setattr(stopwords, "words", counting_words)

    operator = RemoveStopwordsRefiner()
    assert loads == ["english"]

    storage = _storage_with_texts(["The fox", "A dog", "The cat"])
    operator.run(storage, input_key="text")

    assert loads == ["english"]
