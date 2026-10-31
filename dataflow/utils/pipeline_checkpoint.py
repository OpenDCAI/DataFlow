"""Opt-in local JSONL batch commits, isolated from legacy resume cursors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def state_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(" ")
            stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ValueError("Checkpoint directory is already in use") from exc
        try:
            yield
        finally:
            release()


class PipelineCheckpoint:
    def __init__(self, pipeline, directory, batch_size, config, streaming):
        from dataflow.utils.storage import BatchedFileStorage, StreamBatchedFileStorage
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("Enhanced checkpoints require a positive batch_size")
        if not isinstance(config, dict):
            raise ValueError("checkpoint_config must declare prompt/model/operator configuration")
        nodes = [node for node in pipeline.op_nodes_list if node.op_obj is not None]
        if not nodes:
            raise ValueError("Checkpoint pipeline must contain an operator")
        expected_type = StreamBatchedFileStorage if streaming else BatchedFileStorage
        if any(type(node.storage) is not expected_type or node.storage.cache_type != "jsonl" for node in nodes):
            raise ValueError("Enhanced checkpoints support standard local batched JSONL Storage only")
        source = Path(nodes[0].storage.first_entry_file_name).resolve()
        if not source.is_file() or source.suffix.lower() != ".jsonl":
            raise ValueError("Enhanced checkpoints require a local JSONL source")
        if any(Path(node.storage.first_entry_file_name).resolve() != source for node in nodes):
            raise ValueError("Enhanced checkpoints require a single input source")
        self.outputs = [Path(node.storage._get_cache_file_path(node.storage.operator_step + 1)).resolve() for node in nodes]
        if len(set(self.outputs)) != len(self.outputs) or source in self.outputs:
            raise ValueError("Checkpoint outputs must be distinct from each other and the source")
        self.path = Path(directory).resolve() / "manifest.json"
        if self.path in self.outputs or self.path == source:
            raise ValueError("Checkpoint manifest conflicts with data files")
        self.source = source
        self.source_stat = (source.stat().st_size, source.stat().st_mtime_ns, source.stat().st_ino)
        self.source_digest = file_digest(source)
        self.identity = canonical_digest({
            "source": str(source), "sha256": self.source_digest,
            "batch_size": batch_size, "streaming": streaming, "config": config,
            "operators": [{"type": type(n.op_obj).__module__ + "." + type(n.op_obj).__name__, "args": n.kwargs} for n in nodes],
            "outputs": list(map(str, self.outputs)),
            "read_policy": type(nodes[0].storage.jsonl_read_policy).__name__,
        })
        expected_files = {str(path): None for path in self.outputs}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self.state, dict) or not isinstance(self.state.get("files"), dict):
                    raise ValueError("Invalid checkpoint manifest")
                if self.state["version"] != 1 or self.state["identity"] != self.identity:
                    raise ValueError("Input or pipeline configuration changed; use a new checkpoint directory")
                if set(self.state["files"]) != set(expected_files):
                    raise ValueError("Checkpoint output paths do not match")
                for key in ("step", "batch"):
                    if type(self.state[key]) is not int or self.state[key] < 0:
                        raise ValueError("Invalid checkpoint cursor")
                if self.state["step"] > len(nodes) or (self.state["step"] == len(nodes) and self.state["batch"] != 0):
                    raise ValueError("Invalid checkpoint step")
                for size in self.state["files"].values():
                    if size is not None and (type(size) is not int or size < 0):
                        raise ValueError("Invalid committed output size")
                for index, path in enumerate(self.outputs):
                    if (index < self.state["step"] or (index == self.state["step"] and self.state["batch"] > 0)) and self.state["files"][str(path)] is None:
                        raise ValueError("Checkpoint cursor references an uncommitted output")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid checkpoint manifest") from exc
        else:
            if any(path.exists() for path in self.outputs):
                raise ValueError("Output files already exist without an enhanced checkpoint; use a new cache directory")
            self.state = {"version": 1, "identity": self.identity, "step": 0, "batch": 0, "files": expected_files}
            atomic_json(self.path, self.state)
        # Validate every committed file before changing any uncommitted tail.
        for path in self.outputs:
            size = self.state["files"][str(path)]
            if size is not None and (not path.exists() or path.stat().st_size < size):
                raise ValueError(f"Committed output is missing or shorter than checkpoint: {path}")
        for path in self.outputs:
            size = self.state["files"][str(path)]
            if size is None and path.exists():
                path.unlink()
            elif size is not None and path.stat().st_size > size:
                with path.open("r+b") as stream:
                    stream.truncate(size)
                    stream.flush()
                    os.fsync(stream.fileno())

    def commit(self, step, batch):
        stat = self.source.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self.source_stat:
            raise ValueError("Input changed while running; checkpoint was not advanced")
        files = {}
        for path in self.outputs:
            if path.exists():
                # Windows FlushFileBuffers requires a writable handle. r+b
                # preserves committed bytes while allowing fsync on every OS.
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
                files[str(path)] = path.stat().st_size
            else:
                files[str(path)] = None
        state = {**self.state, "step": step, "batch": batch, "files": files}
        atomic_json(self.path, state)
        self.state = state


def run_checkpointed_pipeline(pipeline, directory, batch_size, config, streaming):
    """Commit ordinary operator batches; external side effects are not covered."""
    with state_lock(directory):
        checkpoint = PipelineCheckpoint(pipeline, directory, batch_size, config, streaming)
        resume_step, resume_batch = checkpoint.state["step"], checkpoint.state["batch"]
        resources = []
        try:
            for index, node in enumerate(pipeline.op_nodes_list):
                step = index - 1
                if node.op_obj is None or step < resume_step:
                    continue
                storage = node.storage
                storage.batch_size = batch_size
                start = resume_batch if step == resume_step else 0
                storage._dataframe_buffer.clear()
                if node.llm_serving is not None and all(node.llm_serving is not resource for resource in resources):
                    resources.append(node.llm_serving)
                if streaming:
                    batches = storage.iter_chunks()
                else:
                    storage.batch_step = 0
                    storage.read()
                    total = (storage.record_count + batch_size - 1) // batch_size
                    batches = (None for _ in range(total))
                processed = 0
                for batch, frame in enumerate(batches):
                    processed += 1
                    if batch < start:
                        continue
                    storage.batch_step = batch
                    if streaming:
                        storage._current_streaming_chunk = frame
                    node.op_obj.run(storage=storage, **node.kwargs)
                    if not Path(storage._get_cache_file_path(storage.operator_step + 1)).exists():
                        raise ValueError("Checkpoint operators must write their output through Storage")
                    checkpoint.commit(step, batch + 1)
                    if streaming:
                        storage._current_streaming_chunk = None
                if start > processed:
                    raise ValueError("Checkpoint batch exceeds the current input")
                if processed == 0:
                    # Materialize empty output so downstream steps have an input.
                    storage.batch_step = 0
                    if streaming:
                        storage._current_streaming_chunk = pd.DataFrame()
                    node.op_obj.run(storage=storage, **node.kwargs)
                    if streaming:
                        storage._current_streaming_chunk = None
                if hasattr(node.op_obj, "shutdown"):
                    node.op_obj.shutdown()
                checkpoint.commit(step + 1, 0)
        finally:
            for node in pipeline.op_nodes_list:
                if node.storage is not None and hasattr(node.storage, "_current_streaming_chunk"):
                    node.storage._current_streaming_chunk = None
            for resource in resources:
                resource.cleanup()
