import pandas as pd
import pytest

from dataflow.operators.general_text import RemoveEmoticonsRefiner
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
        return "memory://remove-emoticons-refiner"


def _refine(texts):
    storage = MemoryStorage(pd.DataFrame({"text": texts}))
    RemoveEmoticonsRefiner().run(storage, input_key="text")
    return storage.result["text"].tolist()


@pytest.mark.cpu
def test_removes_emoticons():
    assert _refine(["great :D", "hello:)", "so sad :("]) == ["great ", "hello", "so sad "]


@pytest.mark.cpu
def test_longest_emoticon_is_removed_whole():
    assert _refine(["hi :-)))"]) == ["hi "]


@pytest.mark.cpu
def test_letter_emoticons_do_not_eat_word_endings():
    assert _refine(["Name and: bob", "Read: this; sound: ok"]) == [
        "Name and: bob",
        "Read: this; sound: ok",
    ]
