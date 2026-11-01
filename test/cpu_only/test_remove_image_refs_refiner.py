import pandas as pd
import pytest

from dataflow.operators.general_text import RemoveImageRefsRefiner
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
        return "memory://remove-image-refs-refiner"


def refine(texts):
    storage = MemoryStorage(pd.DataFrame({"text": texts}))
    RemoveImageRefsRefiner().run(storage, input_key="text")
    return storage.result["text"].tolist()


@pytest.mark.cpu
def test_words_of_seven_or_more_characters_are_kept():
    texts = [
        "Machine learning algorithms are powerful.",
        "Information retrieval matters",
        "Install Python3 and TensorFlow2 first.",
    ]

    assert refine(texts) == texts


@pytest.mark.cpu
def test_image_references_are_still_removed():
    texts = [
        "See ![](images/a.jpg) the figure",
        "Intro ![](images/3f2a9c0d1e4b.jpg) end",
        "结论 图 1-2：系统架构",
    ]

    assert refine(texts) == [
        "See  the figure",
        "Intro  end",
        "结论 ",
    ]
