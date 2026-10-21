"""Inject interruptions on either side of the batch commit boundary."""
import json
import os
from pathlib import Path
import pandas as pd
import pytest
from dataflow.core import OperatorABC
from dataflow.pipeline import BatchedPipelineABC, StreamBatchedPipelineABC
from dataflow.utils.pipeline_checkpoint import PipelineCheckpoint
from dataflow.utils.storage import BatchedFileStorage, StreamBatchedFileStorage

pytestmark = pytest.mark.cpu

class Compute(OperatorABC):
    def __init__(self, crash=None, filter_all=False):
        super().__init__()
        self.crash, self.filter_all = crash, filter_all
        self.calls = 0

    def run(self, storage, input_key="number", output_key="result"):
        frame = storage.read().copy()
        frame[output_key] = frame.get(input_key, pd.Series(dtype="int64")) * 2
        if self.filter_all:
            frame = frame.iloc[:0]
        call = self.calls
        self.calls += 1
        if self.crash == (call, "before"):
            raise RuntimeError("injected before write")
        storage.write(frame)
        if self.crash == (call, "after"):
            raise RuntimeError("injected after write")
        return [output_key]

class Prepare(BatchedPipelineABC):
    storage_type = BatchedFileStorage
    def __init__(self, source, root, crash=None, filter_all=False):
        super().__init__()
        self.storage = self.storage_type(str(source), cache_path=str(root / "cache"))
        self.first, self.second = Compute(crash, filter_all), Compute()
    def forward(self):
        self.first.run(self.storage.step())
        self.second.run(self.storage.step(), input_key="result", output_key="final")

class StreamPrepare(Prepare):
    storage_type = StreamBatchedFileStorage
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward

def run(cls, source, root, **kwargs):
    pipeline = cls(source, root, **kwargs)
    pipeline.compile()
    pipeline.forward(batch_size=2, checkpoint_dir=root / "state", checkpoint_config={"revision": 1, "model": "none"})
    path = Path(pipeline.storage._get_cache_file_path(2))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(''.join(json.dumps({"number": i}) + '\n' for i in range(7)))
    return source

@pytest.mark.parametrize("cls", [Prepare, StreamPrepare])
@pytest.mark.parametrize("crash", [(0, "before"), (0, "after"), (1, "after"), (3, "after")])
def test_restart_matches_uninterrupted_execution(tmp_path, source, cls, crash):
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        run(cls, source, tmp_path / "restart", crash=crash)
    actual = run(cls, source, tmp_path / "restart")
    assert actual == run(cls, source, tmp_path / "baseline")
    assert [row["number"] for row in actual] == list(range(7))
    assert [row["final"] for row in actual] == [i * 4 for i in range(7)]
    assert run(cls, source, tmp_path / "restart") == actual
    assert source.read_bytes() == original

@pytest.mark.parametrize("cls", [Prepare, StreamPrepare])
def test_restart_after_manifest_commit(tmp_path, source, cls, monkeypatch):
    original = PipelineCheckpoint.commit
    def interrupt(checkpoint, step, batch):
        original(checkpoint, step, batch)
        if step == 0 and batch == 1:
            raise RuntimeError("injected after commit")
    with monkeypatch.context() as patch:
        patch.setattr(PipelineCheckpoint, "commit", interrupt)
        with pytest.raises(RuntimeError, match="after commit"):
            run(cls, source, tmp_path / "restart")
    assert run(cls, source, tmp_path / "restart") == run(cls, source, tmp_path / "baseline")

@pytest.mark.parametrize("cls", [Prepare, StreamPrepare])
def test_empty_filtered_batches_can_resume(tmp_path, source, cls):
    assert run(cls, source, tmp_path / "run", filter_all=True) == []
    assert run(cls, source, tmp_path / "run", filter_all=True) == []

def test_input_configuration_and_batch_changes_rejected(tmp_path, source):
    root = tmp_path / "run"
    run(Prepare, source, root)
    pipeline = Prepare(source, root)
    pipeline.compile()
    with pytest.raises(ValueError, match="changed"):
        pipeline.forward(batch_size=2, checkpoint_dir=root / "state", checkpoint_config={"revision": 2})
    with pytest.raises(ValueError, match="changed"):
        pipeline.forward(batch_size=3, checkpoint_dir=root / "state", checkpoint_config={"revision": 1, "model": "none"})
    source.write_text(source.read_text() + '{"number":9}\n')
    with pytest.raises(ValueError, match="changed"):
        run(Prepare, source, root)

@pytest.mark.parametrize("damage", ["missing", "short", "invalid_json", "invalid_cursor", "invalid_end_step", "completed_batch"])
def test_damaged_state_or_output_does_not_silently_restart(tmp_path, source, damage):
    root = tmp_path / "run"
    run(Prepare, source, root)
    output = root / "cache/dataflow_cache_step_step2.jsonl"
    manifest = root / "state/manifest.json"
    if damage == "missing":
        output.unlink()
    elif damage == "short":
        output.write_text('')
    elif damage == "invalid_json":
        manifest.write_text('{')
    else:
        state = json.loads(manifest.read_text())
        if damage == "invalid_end_step":
            state["step"] = 3  # Two operators: only steps 0, 1 and completed 2 exist.
        elif damage == "completed_batch":
            state["batch"] = 1  # A completed pipeline cannot have an active batch.
        else:
            state["batch"] = -1
        manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        run(Prepare, source, root)

def test_legacy_outputs_require_separate_directory(tmp_path, source):
    pipeline = Prepare(source, tmp_path)
    pipeline.compile()
    pipeline.forward(batch_size=2, resume_from_last=False)
    with pytest.raises(ValueError, match="already exist"):
        run(Prepare, source, tmp_path)


def test_commits_flush_writable_handles_without_changing_output(tmp_path, source, monkeypatch):
    """Emulate Windows' writable-handle rule, even when running on POSIX."""
    original_open, original_fsync = Path.open, os.fsync
    handles, flushed = {}, []

    def track_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.suffix == ".jsonl" and path.parent.name == "cache":
            handles[stream.fileno()] = stream
        return stream

    def writable_fsync(descriptor):
        stream = handles.get(descriptor)
        if stream is not None and not stream.closed:
            if not stream.writable():
                raise OSError("FlushFileBuffers requires a writable file handle")
            flushed.append(stream.name)
        return original_fsync(descriptor)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(os, "fsync", writable_fsync)
    actual = run(Prepare, source, tmp_path / "run")
    assert flushed
    assert [row["final"] for row in actual] == [i * 4 for i in range(7)]


def test_unicode_checkpoint_paths_resume_with_non_utf8_platform_default(tmp_path, source, monkeypatch):
    original = Path.read_text

    def windows_read(path, encoding=None, errors=None):
        return original(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_read)
    root = tmp_path / "中文状态"
    first = run(Prepare, source, root)
    assert run(Prepare, source, root) == first
