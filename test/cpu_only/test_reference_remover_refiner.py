import pandas as pd
import pytest

from dataflow.operators.general_text import ReferenceRemoverRefiner
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
        return "memory://reference-remover-refiner"


@pytest.mark.cpu
def test_reference_remover_refiner_writes_refined_text():
    storage = MemoryStorage(pd.DataFrame({"text": [
        "Paris is the capital of France.<ref>Smith 2020</ref>",
        "Water boils at 100 C.{{cite web|url=http://example.com}}",
        "No references here.",
    ]}))

    ReferenceRemoverRefiner().run(storage, input_key="text")

    assert storage.result["text"].tolist() == [
        "Paris is the capital of France.",
        "Water boils at 100 C.",
        "No references here.",
    ]
