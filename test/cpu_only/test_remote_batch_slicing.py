import sys
import types
from unittest.mock import Mock

import pandas as pd
import pytest

from dataflow.utils.storage import BatchedFileStorage, StreamBatchedFileStorage


def mock_remote_source(monkeypatch, source_kind, dataframe):
    if source_kind == "hf":
        loader = Mock(return_value=types.SimpleNamespace(to_pandas=dataframe.copy))
        module = types.ModuleType("datasets")
        module.load_dataset = loader
        monkeypatch.setitem(sys.modules, "datasets", module)
        source = "hf:example/data:subset:validation"
    else:
        loader = Mock(return_value=dataframe.to_dict(orient="records"))
        module = types.ModuleType("modelscope")
        module.MsDataset = types.SimpleNamespace(load=loader)
        monkeypatch.setitem(sys.modules, "modelscope", module)
        source = "ms:example/data:validation"
    return source, loader


@pytest.mark.cpu
@pytest.mark.parametrize("source_kind", ["hf", "ms"])
@pytest.mark.parametrize("output_type", ["dataframe", "dict"])
def test_remote_batches_are_sliced_counted_and_cached(
    monkeypatch, tmp_path, source_kind, output_type
):
    source, loader = mock_remote_source(
        monkeypatch,
        source_kind,
        pd.DataFrame({"text": ["first", "second", "third"]}, index=[10, 20, 30]),
    )
    storage = BatchedFileStorage(source, cache_path=str(tmp_path)).step()
    storage.batch_size = 2

    for batch_step, expected in enumerate([["first", "second"], ["third"], []]):
        storage.batch_step = batch_step
        result = storage.read(output_type)
        if output_type == "dataframe":
            assert result["text"].tolist() == expected
            assert result.index.tolist() == list(range(len(expected)))
            if not result.empty:
                result.loc[0, "text"] = "modified by caller"
        else:
            assert result == [{"text": text} for text in expected]
            if result:
                result[0]["text"] = "modified by caller"
        assert storage.record_count == 3

    storage.batch_step = 0
    assert storage.read("dict") == [{"text": "first"}, {"text": "second"}]
    if source_kind == "hf":
        loader.assert_called_once_with("example/data", "subset", split="validation")
    else:
        loader.assert_called_once_with("example/data", split="validation")


@pytest.mark.cpu
@pytest.mark.parametrize("source_kind", ["hf", "ms"])
def test_remote_read_without_batch_size_keeps_all_rows(monkeypatch, tmp_path, source_kind):
    source, loader = mock_remote_source(
        monkeypatch, source_kind, pd.DataFrame({"text": ["first", "second", "third"]})
    )
    storage = BatchedFileStorage(source, cache_path=str(tmp_path)).step()

    assert storage.read("dict") == [
        {"text": "first"}, {"text": "second"}, {"text": "third"}
    ]
    assert storage.record_count == 3
    loader.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("source_kind", ["hf", "ms"])
def test_empty_remote_dataset_has_zero_records(monkeypatch, tmp_path, source_kind):
    source, loader = mock_remote_source(monkeypatch, source_kind, pd.DataFrame())
    storage = BatchedFileStorage(source, cache_path=str(tmp_path)).step()
    storage.batch_size = 2

    assert storage.read().empty
    assert storage.read("dict") == []
    assert storage.record_count == 0
    loader.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("storage_class", [BatchedFileStorage, StreamBatchedFileStorage])
@pytest.mark.parametrize("extension", ["jsonl", "csv"])
def test_local_batch_reads_keep_slicing_and_caching(tmp_path, storage_class, extension):
    source = tmp_path / f"source.{extension}"
    dataframe = pd.DataFrame({"text": ["first", "second", "third"]})
    if extension == "jsonl":
        dataframe.to_json(source, orient="records", lines=True)
    else:
        dataframe.to_csv(source, index=False)
    storage = storage_class(str(source), cache_path=str(tmp_path)).step()
    storage.batch_size = 2
    storage.batch_step = 1

    result = storage.read()
    assert result["text"].tolist() == ["third"]
    assert result.index.tolist() == [0]
    assert storage.record_count == 3

    # Cached reads must not reload a source that has already been read.
    source.unlink()
    storage.batch_step = 0
    assert storage.read("dict") == [{"text": "first"}, {"text": "second"}]


@pytest.mark.cpu
@pytest.mark.parametrize("source_kind", ["hf", "ms"])
def test_later_step_reads_local_output_without_reloading_remote_source(
    monkeypatch, tmp_path, source_kind
):
    source, loader = mock_remote_source(
        monkeypatch, source_kind, pd.DataFrame({"text": ["first", "second", "third"]})
    )
    storage = BatchedFileStorage(source, cache_path=str(tmp_path)).step()
    storage.read()
    storage.write(pd.DataFrame({"text": ["processed first", "processed second", "processed third"]}))
    next_step = storage.step()
    next_step.batch_size = 2
    next_step.batch_step = 1

    assert next_step.read("dict") == [{"text": "processed third"}]
    assert next_step.record_count == 3
    loader.assert_called_once()
