"""Verify file-format read/write and batch append contracts."""

import pandas as pd
import pytest

from dataflow.utils.storage import (
    BatchedFileStorage,
    FileStorage,
    StreamBatchedFileStorage,
)


pytestmark = pytest.mark.cpu


ROWS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "text": ["a", "b", "c", "d", "e"],
    }
)


def write_source(path, file_type):
    if file_type == "json":
        ROWS.to_json(path, orient="records", force_ascii=False, indent=2)
    elif file_type == "jsonl":
        ROWS.to_json(path, orient="records", lines=True, force_ascii=False)
    elif file_type == "csv":
        ROWS.to_csv(path, index=False)
    elif file_type == "parquet":
        pytest.importorskip("pyarrow")
        ROWS.to_parquet(path, index=False)
    elif file_type == "pickle":
        ROWS.to_pickle(path)
    elif file_type == "xlsx":
        pytest.importorskip("openpyxl")
        ROWS.to_excel(path, index=False)
    else:
        raise AssertionError(file_type)


@pytest.mark.parametrize("file_type", ["json", "jsonl", "csv", "parquet", "pickle", "xlsx"])
def test_file_storage_roundtrip_is_whole_file_replace(tmp_path, file_type):
    source = tmp_path / f"source.{file_type}"
    write_source(source, file_type)
    storage = FileStorage(
        str(source), cache_path=str(tmp_path / "cache"), cache_type=file_type
    ).step()

    assert storage.read("dataframe")["id"].tolist() == [1, 2, 3, 4, 5]
    storage.write(storage.read("dataframe"))

    changed = ROWS.copy()
    changed.loc[0, "id"] = 99
    storage.write(changed)
    output = storage._get_cache_file_path(1)
    reread = FileStorage(
        output, cache_path=str(tmp_path / "reread"), cache_type=file_type
    ).step().read("dataframe")

    assert reread["id"].tolist() == [99, 2, 3, 4, 5]


@pytest.mark.parametrize("storage_cls", [BatchedFileStorage, StreamBatchedFileStorage])
@pytest.mark.parametrize("file_type", ["jsonl", "csv"])
def test_batched_storage_appends_jsonl_and_csv_batches(tmp_path, storage_cls, file_type):
    source = tmp_path / f"source.{file_type}"
    write_source(source, file_type)
    storage = storage_cls(
        str(source), cache_path=str(tmp_path / "cache"), cache_type=file_type
    ).step()
    storage.batch_size = 2

    for batch, frame in enumerate(storage.iter_batches()):
        storage.batch_step = batch
        storage._active_batch = frame
        try:
            storage.write(frame)
        finally:
            storage._active_batch = None

    output = storage._get_cache_file_path(1)
    reread = FileStorage(
        output, cache_path=str(tmp_path / "reread"), cache_type=file_type
    ).step().read("dataframe")
    assert reread["id"].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("storage_cls", [BatchedFileStorage, StreamBatchedFileStorage])
@pytest.mark.parametrize("file_type", ["json", "parquet", "pickle", "xlsx"])
def test_batched_storage_rejects_non_append_formats(tmp_path, storage_cls, file_type):
    with pytest.raises(ValueError, match="only supports 'jsonl' and 'csv'"):
        storage_cls(str(tmp_path / "source.jsonl"), cache_type=file_type)
