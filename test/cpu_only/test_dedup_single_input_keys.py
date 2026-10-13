import pandas as pd
import pytest

from dataflow.operators.general_text import (
    HashDeduplicateFilter,
    MinHashDeduplicateFilter,
    NgramHashDeduplicateFilter,
    SimHashDeduplicateFilter,
)
from dataflow.operators.text_pt import CCNetDeduplicateFilter
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
        return "memory://dedup-single-input-keys"


TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a completely different sentence about something else entirely",
    "the quick brown fox jumps over the lazy dog",
]

OPERATORS = [
    HashDeduplicateFilter,
    MinHashDeduplicateFilter,
    NgramHashDeduplicateFilter,
    SimHashDeduplicateFilter,
    CCNetDeduplicateFilter,
]


@pytest.mark.cpu
@pytest.mark.parametrize("operator_cls", OPERATORS)
def test_single_element_input_keys_matches_input_key(operator_cls):
    dataframe = pd.DataFrame({"text": TEXTS, "other": ["x", "y", "z"]})

    by_key = MemoryStorage(dataframe)
    operator_cls().run(by_key, input_key="text")

    by_keys = MemoryStorage(dataframe)
    operator_cls().run(by_keys, input_keys=["text"])

    assert by_keys.result["text"].tolist() == TEXTS[:2]
    pd.testing.assert_frame_equal(by_keys.result, by_key.result)
