import pandas as pd
import pytest

from dataflow.operators.general_text import TextNormalizationRefiner
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
        return "memory://text-normalization-refiner"


def _refine(texts):
    storage = MemoryStorage(pd.DataFrame({"text": texts}))
    TextNormalizationRefiner().run(storage, input_key="text")
    return storage.result["text"].tolist()


@pytest.mark.cpu
def test_whole_dollar_amount():
    assert _refine(["It costs $5 and $ 20."]) == ["It costs 5 USD and 20 USD."]


@pytest.mark.cpu
def test_thousands_separator_and_decimals_stay_in_the_amount():
    assert _refine(["Paid $1,000 and $12.50 and $1,234,567.89 today"]) == [
        "Paid 1,000 USD and 12.50 USD and 1,234,567.89 USD today"
    ]


@pytest.mark.cpu
def test_trailing_comma_is_not_part_of_the_amount():
    assert _refine(["Prices: $5, $6."]) == ["Prices: 5 USD, 6 USD."]
