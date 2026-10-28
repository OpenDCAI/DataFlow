"""CIDEr corpus mode must estimate document frequencies across all rows."""

import math
import pickle
from collections import defaultdict

import pandas as pd
import pytest

from dataflow.operators.general_text import CiderSampleEvaluator
from dataflow.operators.general_text.eval.cider.cider import precook
from dataflow.utils.storage import DataFlowStorage

pytestmark = pytest.mark.cpu


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://cider"


def corpus_frame():
    return pd.DataFrame(
        {
            "generated": ["one two three four five", "six seven eight nine ten"],
            "reference": ["one two three four five", "six seven eight nine ten"],
            "metadata": ["first", "second"],
        },
        index=[8, 3],
    )


@pytest.mark.parametrize("output_key", ["CiderScore", "similarity"])
def test_corpus_run_scores_exact_matches_and_preserves_row_alignment(output_key):
    source = corpus_frame()
    storage = MemoryStorage(source)

    CiderSampleEvaluator(df_mode="corpus").run(
        storage,
        input_key="generated",
        input_reference_key="reference",
        output_key=output_key,
    )

    # The two disjoint references give every n-gram a nonzero IDF. Exact
    # matches therefore attain the maximum CIDEr score, rather than zero.
    assert storage.result[output_key].tolist() == pytest.approx([10.0, 10.0])
    pd.testing.assert_frame_equal(storage.result[source.columns], source)
    pd.testing.assert_frame_equal(source, corpus_frame())


def test_corpus_eval_returns_each_samples_score_in_input_order():
    source = corpus_frame()
    source.loc[3, "generated"] = "unrelated tokens never appear in references"

    scores = CiderSampleEvaluator(df_mode="corpus").eval(
        source, "generated", "reference"
    )

    assert scores == pytest.approx([10.0, 0.0])


@pytest.mark.parametrize("sigma", [2.0, 6.0, 12.0])
def test_corpus_scores_respect_the_gaussian_length_penalty(sigma):
    source = corpus_frame()
    source.loc[8, "generated"] = "one two three four"

    scores = CiderSampleEvaluator(df_mode="corpus", sigma=sigma).eval(
        source, "generated", "reference"
    )

    # All n-grams occur in one reference, so their IDFs cancel in the cosine.
    # The hypothesis has one fewer n-gram at each order and one fewer bigram.
    cosine_mean = sum(math.sqrt((5 - n) / (6 - n)) for n in range(1, 5)) / 4
    expected = 10 * cosine_mean * math.exp(-1 / (2 * sigma**2))
    assert scores == pytest.approx([expected, 10.0])


def test_corpus_does_not_reward_ngrams_shared_by_every_reference():
    source = pd.DataFrame(
        {
            "generated": [
                "common shared words here blue",
                "common shared words here red",
            ],
            "reference": [
                "common shared words here red",
                "common shared words here blue",
            ],
        }
    )

    scores = CiderSampleEvaluator(df_mode="corpus").eval(
        source, "generated", "reference"
    )

    # Only the shared n-grams match, and their IDF is log(2) - log(2).
    assert scores == pytest.approx([0.0, 0.0])


def test_each_corpus_eval_uses_only_its_own_references():
    evaluator = CiderSampleEvaluator(df_mode="corpus")
    source = corpus_frame()

    assert evaluator.eval(source, "generated", "reference") == pytest.approx(
        [10.0, 10.0]
    )
    # A singleton corpus has zero IDF, even after a previous multi-row call.
    assert evaluator.eval(source.iloc[:1], "generated", "reference") == [0.0]
    assert evaluator.eval(source, "generated", "reference") == pytest.approx(
        [10.0, 10.0]
    )


def test_empty_corpus_writes_an_empty_score_column():
    source = corpus_frame().iloc[:0]
    storage = MemoryStorage(source)

    CiderSampleEvaluator(df_mode="corpus").run(
        storage, input_key="generated", input_reference_key="reference"
    )

    assert storage.result["CiderScore"].tolist() == []
    pd.testing.assert_frame_equal(storage.result[source.columns], source)


def test_precomputed_idf_remains_independent_of_evaluation_corpus(tmp_path):
    source = corpus_frame()
    reference = source.iloc[0]["reference"]
    # A valid external corpus with two documents, one containing these n-grams.
    idf = {
        "df": defaultdict(float, dict.fromkeys(precook(reference), 1.0)),
        "ref_len": 2,
    }
    idf_path = tmp_path / "document-frequency.p"
    with idf_path.open("wb") as handle:
        pickle.dump(idf, handle)
    evaluator = CiderSampleEvaluator(idf_path=idf_path)

    # Unlike local corpus mode, an external IDF also scores a singleton.
    assert evaluator.eval(source.iloc[:1], "generated", "reference") == pytest.approx(
        [10.0]
    )
    assert evaluator.eval(source, "generated", "reference") == pytest.approx(
        [10.0, 10.0]
    )
