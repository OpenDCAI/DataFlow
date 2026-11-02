import pandas as pd
import pytest

from dataflow.operators.code import CodeFileTypeContentFilter
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
        return "memory://code-file-type-content-filter"


def kept_filenames(rows):
    storage = MemoryStorage(pd.DataFrame(rows))
    CodeFileTypeContentFilter().run(storage, input_key="dataframe")
    return storage.result["filename"].tolist()


@pytest.mark.cpu
def test_text_files_follow_filename_rule():
    rows = {
        "filetype": ["Text", "Text", "Text"],
        "filename": ["README.md", "random_dump.txt", "requirements.txt"],
        "line_count": [10, 10, 10],
    }
    assert kept_filenames(rows) == ["README.md", "requirements.txt"]


@pytest.mark.cpu
def test_text_files_still_need_size_check():
    rows = {
        "filetype": ["Text", "Text"],
        "filename": ["notes.txt", "todo.txt"],
        "line_count": [10, 1000],
    }
    assert kept_filenames(rows) == ["notes.txt"]


@pytest.mark.cpu
def test_other_size_checked_types_unchanged():
    rows = {
        "filetype": ["JSON", "JSON", "Python"],
        "filename": ["data.json", "big.json", "main.py"],
        "line_count": [10, 1000, 1000],
    }
    assert kept_filenames(rows) == ["data.json", "main.py"]
