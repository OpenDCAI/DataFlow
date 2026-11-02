import os
import tempfile

import pandas as pd
import pytest

from dataflow.utils.storage import LazyFileStorage


@pytest.fixture()
def storage():
    root = tempfile.mkdtemp()
    source = os.path.join(root, "input.jsonl")
    pd.DataFrame({"text": ["a", "b"]}).to_json(
        source, orient="records", lines=True, force_ascii=False
    )
    return LazyFileStorage(
        first_entry_file_name=source,
        cache_path=os.path.join(root, "cache"),
        file_name_prefix="cache",
        cache_type="jsonl",
        save_on_exit=False,
    )


@pytest.mark.cpu
def test_write_accepts_a_dataframe(storage):
    first = storage.step()
    path = first.write(pd.DataFrame({"text": ["x"]}))

    assert path.endswith("cache_step1.jsonl")
    assert storage.step().read()["text"].tolist() == ["x"]


@pytest.mark.cpu
def test_write_accepts_a_list_of_dicts(storage):
    first = storage.step()
    first.write([{"text": "x"}, {"text": "y"}])

    assert storage.step().read()["text"].tolist() == ["x", "y"]
