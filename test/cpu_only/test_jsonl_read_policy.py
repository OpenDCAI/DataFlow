"""Quarantine must agree across file, lazy, batched and streaming readers."""

import json
from pathlib import Path

import pandas as pd
import pytest

from dataflow.operators.text_sft import QAMessagesGenerator
from dataflow.operators.conversations import ConversationStructureSampleEvaluator
from dataflow.pipeline import PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC
from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.storage import FileStorage, LazyFileStorage, BatchedFileStorage, StreamBatchedFileStorage

pytestmark = pytest.mark.cpu


def make_source(tmp_path):
    source = tmp_path / "input.jsonl"
    lines = [b'broken\n']
    for index in range(1, 99):
        lines.append((json.dumps({"id": index, "q": f"问题{index}", "a": "answer"}, ensure_ascii=False) + "\n").encode())
    lines[50] = b'{\n'
    lines.append(b'\xff\n')
    source.write_bytes(b''.join(lines))
    return source, lines


@pytest.mark.parametrize("cls", [FileStorage, LazyFileStorage, BatchedFileStorage, StreamBatchedFileStorage])
def test_100_records_97_accepted_and_three_errors(tmp_path, cls):
    source, lines = make_source(tmp_path)
    policy = JSONLReadPolicy(tmp_path / "errors.jsonl")
    kwargs = {"save_on_exit": False} if cls is LazyFileStorage else {}
    storage = cls(str(source), cache_path=str(tmp_path / "cache"), jsonl_read_policy=policy, **kwargs).step()
    assert storage.get_keys_from_dataframe() == ["id", "q", "a"]
    if cls is StreamBatchedFileStorage:
        storage.batch_size = 7
        assert storage.get_record_count() == 97
        chunks = list(storage.iter_chunks())
        assert max(map(len, chunks)) <= 7
        rows = [row for frame in chunks for row in frame.to_dict("records")]
        locations = [item for frame in chunks for item in frame.attrs["dataflow_source_locations"]]
    else:
        frame = storage.read()
        rows = frame.to_dict("records")
        locations = frame.attrs["dataflow_source_locations"]
    assert [row["id"] for row in rows] == [i for i in range(1, 99) if i != 50]
    assert (locations[0]["line"], locations[0]["byte_offset"]) == (2, len(lines[0]))
    assert locations[-1]["line"] == 99
    ledger = [json.loads(line) for line in policy.errors_path.read_text().splitlines()]
    assert [error["line"] for error in ledger] == [1, 51, 100]
    assert [error["code"] for error in ledger] == ["invalid_json", "invalid_json", "invalid_utf8"]
    assert ledger[1]["byte_offset"] == sum(map(len, lines[:50]))
    assert policy.last_report["scanned"] == 100
    assert policy.last_report["accepted"] == 97
    assert policy.last_report["rejected"] == 3
    # Header/count/data reads must not append duplicate ledger records.
    storage.read()
    assert len(policy.errors_path.read_text().splitlines()) == 3
    assert source.read_bytes() == b''.join(lines)


@pytest.mark.parametrize("raw,code", [
    (b'\n', "blank_line"), (b'[]\n', "not_object"),
    (b'null\n', "not_object"), (b'{"x": NaN}\n', "invalid_json"),
])
def test_non_records_are_audited_and_empty_source_is_supported(tmp_path, raw, code):
    source = tmp_path / "input.jsonl"
    source.write_bytes(raw)
    policy = JSONLReadPolicy(tmp_path / "errors.jsonl")
    assert policy.read(source).empty
    assert policy.last_report["rejected"] == 1
    assert json.loads(policy.errors_path.read_text())["code"] == code
    source.write_bytes(b'')
    assert policy.read(source).empty
    assert policy.last_report["scanned"] == 0
    assert policy.errors_path.read_text() == ""


def test_missing_file_and_filesystem_errors_are_not_bad_records(tmp_path, monkeypatch):
    policy = JSONLReadPolicy(tmp_path / "errors.jsonl")
    source = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        policy.read(source)
    source.write_text('{}\n')
    original = Path.open
    def deny(path, *args, **kwargs):
        if path == source.resolve():
            raise PermissionError("test denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(PermissionError):
        policy.read(source)
    assert not policy.errors_path.exists()


def test_default_reader_still_rejects_bad_json(tmp_path):
    source, _ = make_source(tmp_path)
    with pytest.raises(ValueError):
        FileStorage(str(source)).step().read()


def test_error_destination_cannot_replace_source_or_cache(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{}\n')
    with pytest.raises(ValueError, match="source"):
        FileStorage(str(source), jsonl_read_policy=JSONLReadPolicy(source))
    with pytest.raises(ValueError, match="cache"):
        FileStorage(str(source), cache_path=str(tmp_path), jsonl_read_policy=JSONLReadPolicy(tmp_path / "dataflow_cache_step_step1.jsonl"))


class Prepare(PipelineABC):
    def __init__(self, storage):
        super().__init__()
        self.storage = storage
        self.convert = QAMessagesGenerator()
        self.validate = ConversationStructureSampleEvaluator()

    def forward(self):
        self.convert.run(self.storage.step(), input_key="q", input_answer_key="a")
        self.validate.run(self.storage.step(), input_key="messages")


class BatchedPrepare(Prepare, BatchedPipelineABC):
    _compiled_forward = BatchedPipelineABC._compiled_forward


class StreamPrepare(Prepare, StreamBatchedPipelineABC):
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward


@pytest.mark.parametrize("cls,pipeline_cls", [
    (FileStorage, Prepare), (LazyFileStorage, Prepare),
    (BatchedFileStorage, BatchedPrepare), (StreamBatchedFileStorage, StreamPrepare),
])
def test_pipeline_compile_and_output_with_bad_first_line(tmp_path, cls, pipeline_cls):
    source, _ = make_source(tmp_path)
    kwargs = {"save_on_exit": False} if cls is LazyFileStorage else {}
    storage = cls(str(source), cache_path=str(tmp_path / "cache"), jsonl_read_policy=JSONLReadPolicy(tmp_path / "errors.jsonl"), **kwargs)
    pipeline = pipeline_cls(storage)
    pipeline.compile()
    assert storage.jsonl_read_policy.inspect(source)["keys"] == ["id", "q", "a"]
    if cls in (BatchedFileStorage, StreamBatchedFileStorage):
        pipeline.forward(batch_size=8, resume_from_last=False)
    else:
        pipeline.forward()
    if cls is LazyFileStorage:
        storage.flush_all()
    actual = [json.loads(line) for line in Path(storage._get_cache_file_path(2)).read_text().splitlines()]
    assert len(actual) == 97
    assert [row["id"] for row in actual] == [i for i in range(1, 99) if i != 50]


def test_evaluation_error_keeps_physical_source_line(tmp_path):
    from dataflow.operators.text_sft import QAStructureSampleEvaluator
    source = tmp_path / 'input.jsonl'
    source.write_text('bad\n{"prompt":"q","answer":""}\n')
    storage = FileStorage(str(source), jsonl_read_policy=JSONLReadPolicy(tmp_path/'reader-errors')).step()
    QAStructureSampleEvaluator().run(storage)
    frame = pd.read_json(storage._get_cache_file_path(1), lines=True)
    error = frame.iloc[0]['source_error']
    assert error['source_location']['line'] == 2
    assert error['field'] == 'answer'
    assert len(frame) == 1
