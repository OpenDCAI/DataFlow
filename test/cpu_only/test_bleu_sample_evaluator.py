import pandas as pd
import pytest

from dataflow.operators.general_text.eval.bleu_sample_evaluator import BleuSampleEvaluator
from dataflow.operators.general_text.eval.bleu.bleu import Bleu
from dataflow.utils.storage import DataFlowStorage


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
        return "memory://bleu-sample-evaluator"


REFERENCE = "the cat sat on the mat"
SHUFFLED = "mat the on sat cat the"


@pytest.mark.cpu
def test_default_score_is_bleu_4_not_unigram_precision():
    storage = MemoryStorage(pd.DataFrame({
        "generated": [REFERENCE, SHUFFLED],
        "reference": [REFERENCE, REFERENCE],
    }))

    BleuSampleEvaluator().run(storage, input_key="generated", input_reference_key="reference")

    identical, shuffled = storage.result["BleuScore"].tolist()
    assert identical == pytest.approx(1.0)
    assert shuffled == pytest.approx(0.0, abs=1e-6)


@pytest.mark.cpu
def test_unigram_bleu_ignores_word_order():
    evaluator = BleuSampleEvaluator(n=1)

    assert evaluator._score_func(SHUFFLED, REFERENCE) == pytest.approx(1.0)


@pytest.mark.cpu
def test_run_uses_the_configured_ngram_order():
    storage = MemoryStorage(pd.DataFrame({
        "generated": [SHUFFLED],
        "reference": [REFERENCE],
    }))

    BleuSampleEvaluator(n=2).run(
        storage,
        input_key="generated",
        input_reference_key="reference",
    )

    assert storage.result["BleuScore"].tolist() == pytest.approx([0.0], abs=1e-6)


@pytest.mark.cpu
@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_identical_text_scores_one_for_each_ngram_order(n):
    evaluator = BleuSampleEvaluator(n=n)

    assert evaluator._score_func(REFERENCE, REFERENCE) == pytest.approx(1.0)


@pytest.mark.cpu
@pytest.mark.parametrize("eff", ["shortest", "average", "longest"])
def test_every_documented_eff_option_is_supported(eff):
    evaluator = BleuSampleEvaluator(eff=eff)

    assert evaluator._score_func(REFERENCE, REFERENCE) == pytest.approx(1.0)


@pytest.mark.cpu
def test_longest_reference_length_is_used_for_multiple_references():
    scorer = Bleu(
        test="one two",
        refs=["one two", "one two three four"],
        n=2,
    )

    scorer.compute_score(option="longest")

    assert scorer.reflen() == 4
