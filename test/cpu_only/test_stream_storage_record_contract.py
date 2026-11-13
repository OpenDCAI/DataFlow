"""Streaming storage must expose batches and metadata as logical records."""

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import StreamBatchedPipelineABC
from dataflow.utils.storage import StreamBatchedFileStorage


pytestmark = pytest.mark.cpu


def write_source(path, frame):
    extension = path.suffix.lower()
    if extension == ".json":
        frame.to_json(path, orient="records")
    elif extension == ".pickle":
        frame.to_pickle(path)
    elif extension == ".parquet":
        pytest.importorskip("pyarrow")
        frame.to_parquet(path, index=False)
    elif extension == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    else:
        frame.to_csv(path, index=False)


def storage_for(path, tmp_path):
    return StreamBatchedFileStorage(
        str(path), cache_path=str(tmp_path / "cache")
    ).step()


@pytest.mark.parametrize("extension", ["json", "pickle", "parquet", "csv", "jsonl"])
@pytest.mark.parametrize("uppercase", [False, True], ids=["lowercase", "uppercase"])
def test_batches_headers_and_counts_describe_the_same_records(tmp_path, extension, uppercase):
    frame = pd.DataFrame({"value": [1, 2, 3], "text": ["one", "two", "three"]})
    source = tmp_path / f"source.{extension.upper() if uppercase else extension}"
    write_source(source, frame)
    storage = storage_for(source, tmp_path)
    storage.batch_size = 2

    batches = list(storage.iter_batches())
    assert all(isinstance(batch, pd.DataFrame) for batch in batches)
    assert [len(batch) for batch in batches] == [2, 1]
    assert storage.get_keys_from_dataframe() == ["value", "text"]
    assert storage.get_record_count() == 3
    assert pd.concat(batches).to_dict(orient="records") == frame.to_dict(orient="records")
    assert storage.read("dataframe").to_dict(orient="records") == frame.iloc[:2].to_dict(orient="records")


@pytest.mark.parametrize("size", [None, 0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(tmp_path, size):
    source = tmp_path / "source.json"
    write_source(source, pd.DataFrame({"value": [1]}))
    storage = storage_for(source, tmp_path)
    storage.batch_size = size
    with pytest.raises(ValueError, match="batch_size"):
        list(storage.iter_batches())


def test_parquet_header_contains_top_level_fields(tmp_path):
    source = tmp_path / "source.parquet"
    write_source(source, pd.DataFrame({"nested": [{"leaf": 1}], "value": [2]}))
    assert storage_for(source, tmp_path).get_keys_from_dataframe() == ["nested", "value"]


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([10, 20]),
        pd.Index([10, 20], name="row_id"),
        pd.MultiIndex.from_tuples(
            [("a", 10), ("b", 20)], names=["group", "row_id"]
        ),
    ],
    ids=["unnamed", "named", "multiindex"],
)
def test_parquet_header_excludes_persisted_index(tmp_path, index):
    pytest.importorskip("pyarrow")
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame({"value": [1, 2]}, index=index)
    frame.to_parquet(source)
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1

    assert storage.get_keys_from_dataframe() == frame.columns.tolist()
    assert all(batch.columns.tolist() == ["value"] for batch in storage.iter_batches())
    index_key = index.names[-1] or "__index_level_0__"
    pipeline = StreamRecordsPipeline(source, tmp_path / "pipeline", input_key=index_key)
    with pytest.raises(KeyError, match=index_key):
        pipeline.compile()


def test_csv_count_uses_logical_records_and_skips_blank_lines(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text('value,text\n1,"first\ncontinued"\n\n2,second\n', encoding="utf-8")
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    assert storage.get_record_count() == 2
    assert sum(len(frame) for frame in storage.iter_batches()) == 2


@pytest.mark.parametrize("contents", ["", " \n\t\n", "value,text\n"])
def test_empty_csv_has_zero_records(tmp_path, contents):
    source = tmp_path / "source.csv"
    source.write_text(contents, encoding="utf-8")
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    assert storage.get_record_count() == 0
    expected_keys = ["value", "text"] if contents.startswith("value") else []
    assert storage.get_keys_from_dataframe() == expected_keys
    assert list(storage.iter_batches()) == []
    assert storage.read("dataframe").empty


def test_malformed_csv_is_not_treated_as_empty(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text('value,text\n1,"unterminated\n', encoding="utf-8")
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    with pytest.raises(ValueError):
        list(storage.iter_batches())
    with pytest.raises(pd.errors.ParserError):
        storage.get_record_count()


@pytest.mark.parametrize("prefix", ["", "\n", "\n \t\n"])
def test_jsonl_metadata_skips_blank_lines_like_reader(tmp_path, prefix):
    source = tmp_path / "source.jsonl"
    source.write_text(prefix + '{"value": 1}\n\n{"value": 2}\n', encoding="utf-8")
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    assert storage.get_keys_from_dataframe() == ["value"]
    assert storage.get_record_count() == 2
    batches = list(storage.iter_batches())
    assert all(len(frame) for frame in batches)
    assert sum(len(frame) for frame in batches) == 2


def test_closing_batch_iterator_closes_csv_reader(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    write_source(source, pd.DataFrame({"value": [1, 2, 3]}))
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    readers = []
    read_csv = pd.read_csv

    def capture_reader(*args, **kwargs):
        reader = read_csv(*args, **kwargs)
        readers.append(reader)
        return reader

    monkeypatch.setattr(pd, "read_csv", capture_reader)
    batches = storage.iter_batches()
    assert next(batches)["value"].tolist() == [1]
    assert not readers[0].handles.handle.closed
    batches.close()
    assert readers[0].handles.handle.closed


def test_source_symlink_suffix_controls_format(tmp_path):
    target = tmp_path / "records"
    pd.DataFrame({"value": [1, 2]}).to_csv(target, index=False)
    source = tmp_path / "source.csv"
    try:
        source.symlink_to(target)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable")
    storage = storage_for(source, tmp_path)
    storage.batch_size = 1
    assert storage.get_record_count() == 2
    assert sum(len(frame) for frame in storage.iter_batches()) == 2


def test_count_changes_when_current_storage_step_changes(tmp_path):
    source = tmp_path / "source.csv"
    write_source(source, pd.DataFrame({"value": [1, 2, 3]}))
    storage = storage_for(source, tmp_path)
    assert storage.get_record_count() == 3
    storage.write(pd.DataFrame({"value": [1]}))
    storage.step()
    assert storage.get_record_count() == 1


def test_count_cache_is_invalidated_after_same_file_changes(tmp_path):
    source = tmp_path / "source.csv"
    write_source(source, pd.DataFrame({"value": [1, 2]}))
    storage = storage_for(source, tmp_path)
    assert storage.get_record_count() == 2
    pd.DataFrame({"value": [3]}).to_csv(source, index=False, header=False, mode="a")
    assert storage.get_record_count() == 3


class ScaleRecords(OperatorABC):
    def __init__(self):
        super().__init__()
        self.batch_lengths = []

    def run(self, storage, input_key="value", output_key="result"):
        frame = storage.read("dataframe").copy()
        self.batch_lengths.append(len(frame))
        frame[output_key] = frame[input_key] * 2
        storage.write(frame)


class StreamRecordsPipeline(StreamBatchedPipelineABC):
    def __init__(self, source, cache, input_key="value"):
        super().__init__()
        self.storage = StreamBatchedFileStorage(str(source), cache_path=str(cache))
        self.scale = ScaleRecords()
        self.input_key = input_key

    def forward(self):
        self.scale.run(self.storage.step(), input_key=self.input_key, output_key="result")


@pytest.mark.parametrize("extension", ["json", "pickle", "parquet", "jsonl"])
def test_compiled_stream_pipeline_accepts_source(tmp_path, extension):
    source = tmp_path / f"source.{extension}"
    write_source(source, pd.DataFrame({"value": [1, 2, 3]}))
    if extension == "jsonl":
        source.write_text("\n \t\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
    pipeline = StreamRecordsPipeline(source, tmp_path / "cache")
    scale = pipeline.scale
    pipeline.compile()
    pipeline.forward(batch_size=2, resume_from_last=False)
    output = pd.read_json(tmp_path / "cache" / "dataflow_cache_step_step1.jsonl", lines=True)
    assert output["value"].tolist() == [1, 2, 3]
    assert output["result"].tolist() == [2, 4, 6]
    assert scale.batch_lengths == [2, 1]
