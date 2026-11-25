import pandas as pd
import pytest

from dataflow.operators.general_text import NgramHashDeduplicateFilter
from dataflow.utils.storage import DataFlowStorage


class MemoryStorage(DataFlowStorage):
    def __init__(self, rows):
        self.dataframe = pd.DataFrame(rows)
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://ngramhash-chunk-coverage"


def _run(rows, input_key="text", **kwargs):
    storage = MemoryStorage(rows)
    if input_key is None:
        result_keys = NgramHashDeduplicateFilter(**kwargs).run(
            storage, input_keys=["title", "body"]
        )
    else:
        result_keys = NgramHashDeduplicateFilter(**kwargs).run(
            storage, input_key=input_key
        )
    return storage, result_keys


@pytest.mark.cpu
@pytest.mark.parametrize(
    "texts, ngram",
    [
        # 10 characters, 3 chunks: the old code hashed only the first 9 characters.
        (["abcdefghij", "abcdefghix"], 3),
        # 7 characters, 4 chunks: the old code hashed only the first 4 characters.
        (["abcdefg", "abcdefx"], 4),
        # 11 characters, 5 chunks: the old code hashed only the first 10 characters.
        (["abcdefghijk", "abcdefghijx"], 5),
    ],
)
def test_texts_differing_only_in_the_tail_are_not_duplicates(texts, ngram):
    # diff_size == ngram requires every chunk to match, so the trailing characters
    # are the only thing that can tell these two texts apart.
    storage, result_keys = _run([{"text": t} for t in texts], n_gram=ngram, diff_size=ngram)

    assert storage.result["text"].tolist() == texts
    assert storage.result[result_keys[0]].tolist() == [1, 1]


@pytest.mark.cpu
@pytest.mark.parametrize("ngram", [2, 3, 5, 8])
def test_texts_shorter_than_n_gram_stay_distinct(ngram):
    texts = ["a", "b", "c"]

    storage, _ = _run([{"text": t} for t in texts], n_gram=ngram)

    assert storage.result["text"].tolist() == texts


@pytest.mark.cpu
def test_exact_duplicates_are_still_removed():
    storage, result_keys = _run(
        [{"text": t} for t in ["abcdefghij", "abcdefghij", "abcdefghij"]]
    )

    assert storage.result["text"].tolist() == ["abcdefghij"]
    assert storage.result.index.tolist() == [0]
    assert result_keys == ["minhash_deduplicated_label"]


@pytest.mark.cpu
def test_empty_texts_are_deduplicated_into_a_single_row():
    storage, _ = _run([{"text": t} for t in ["", "abcdefghij", "", "abcdefghij"]])

    assert storage.result["text"].tolist() == ["", "abcdefghij"]


@pytest.mark.cpu
def test_chunk_overlap_below_diff_size_is_kept():
    # 9 characters with n_gram=3 -> three chunks of three characters, of which two
    # are shared. diff_size=2 rejects the pair, diff_size=3 tolerates it.
    rows = [{"text": "abcdefghi"}, {"text": "abcXXXghi"}]

    strict, _ = _run(rows, n_gram=3, diff_size=2)
    loose, _ = _run(rows, n_gram=3, diff_size=3)

    assert strict.result["text"].tolist() == ["abcdefghi"]
    assert loose.result["text"].tolist() == ["abcdefghi", "abcXXXghi"]


@pytest.mark.cpu
def test_n_gram_one_hashes_the_whole_text():
    storage, _ = _run(
        [{"text": t} for t in ["abcdefghij", "abcdefghix", "abcdefghij"]], n_gram=1
    )

    assert storage.result["text"].tolist() == ["abcdefghij", "abcdefghix"]


@pytest.mark.cpu
def test_multiple_input_keys_cover_the_tail_of_the_joined_text():
    rows = [
        {"title": "abcdefghij", "body": "klmnopqrst"},
        {"title": "abcdefghij", "body": "klmnopqrsX"},
    ]

    storage, _ = _run(rows, input_key=None, n_gram=3, diff_size=3)

    assert len(storage.result) == 2


@pytest.mark.cpu
def test_large_distinct_batch_keeps_every_row():
    texts = [f"doc-{i:06d}" for i in range(2000)]

    storage, _ = _run([{"text": t} for t in texts], n_gram=1)

    assert len(storage.result) == len(texts)


@pytest.mark.cpu
@pytest.mark.parametrize("bad_kwargs", [{"n_gram": 0}, {"diff_size": 0}])
def test_invalid_chunk_configuration_is_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        NgramHashDeduplicateFilter(**bad_kwargs)
