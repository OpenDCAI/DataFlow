"""P07.01 — production-boundary tests for opt-in stream recovery."""
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import StreamBatchedPipelineABC
from dataflow.utils.storage import StreamBatchedFileStorage
from dataflow.utils.stream_transaction import TransactionError


class _Pass(OperatorABC):
    transaction_safe = True

    def __init__(self, fail=None, output_empty=False, fail_commit=None):
        super().__init__()
        self.fail = fail
        self.output_empty = output_empty
        self.fail_commit = fail_commit
        self.failed = False

    def transaction_signature(self):
        return {"empty": self.output_empty}

    def run(self, storage):
        data = storage.read("dataframe")
        records = data["id"].tolist() if "id" in data else []
        if self.fail == "before" and records == [2, 3] and not self.failed:
            self.failed = True
            raise RuntimeError("before write")
        storage.write([] if self.output_empty else data)
        if self.fail == "after" and records == [2, 3] and not self.failed:
            self.failed = True
            raise RuntimeError("after write")


class _Double(OperatorABC):
    transaction_safe = True

    def transaction_signature(self):
        return {}

    def run(self, storage):
        data = storage.read("dataframe")
        storage.write(data)
        try:
            storage.write(data)
        except TransactionError:
            pass  # Suppressed write errors must still abort the transaction.


class _Pipeline(StreamBatchedPipelineABC):
    def __init__(self, source, cache, fmt, ops):
        super().__init__()
        self.storage = StreamBatchedFileStorage(str(source), str(cache), "p07_01", fmt)
        for i, operator in enumerate(ops):
            setattr(self, "op" + str(i), operator)

    def forward(self):
        for i in range(len([x for x in vars(self) if x.startswith("op") and x[2:].isdigit()])):
            getattr(self, "op" + str(i)).run(self.storage.step())


def _source(tmp_path, fmt, rows=5):
    source = tmp_path / ("source." + fmt)
    data = pd.DataFrame({"id": list(range(rows))})
    if fmt == "csv":
        data.to_csv(source, index=False)
    else:
        data.to_json(source, orient="records", lines=True)
    return source


def _output(path, fmt):
    if not path.exists() or path.stat().st_size == 0:
        return []
    if fmt == "csv":
        return pd.read_csv(path)["id"].tolist()
    return pd.read_json(path, lines=True)["id"].tolist()


def _run(source, cache, fmt, ops):
    p = _Pipeline(source, cache, fmt, ops)
    p.compile()
    p.forward(batch_size=2, transactional_resume=True)


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
@pytest.mark.parametrize("fail", ["before", "after"])
def test_transaction_replay_on_same_instance(tmp_path, fmt, fail):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    op = _Pass(fail=fail)
    pipeline = _Pipeline(source, cache, fmt, [op])
    pipeline.compile()
    with pytest.raises(RuntimeError, match=fail + " write"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    pipeline.forward(batch_size=2, transactional_resume=True)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_transaction_replay_after_new_pipeline(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        _run(source, cache, fmt, [_Pass(fail="after")])
    _run(source, cache, fmt, [_Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_two_stages_and_stage_resume(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        _run(source, cache, fmt, [_Pass(), _Pass(fail="after")])
    _run(source, cache, fmt, [_Pass(), _Pass()])
    for n in (1, 2):
        assert _output(cache / (f"p07_01_step{n}." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_logical_empty_stage(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_Pass(output_empty=True), _Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == []
    assert not (cache / ("p07_01_step2." + fmt)).exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_multiple_writes_even_if_caught_fails(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    with pytest.raises(TransactionError, match="one successful write"):
        _run(source, tmp_path / "cache", fmt, [_Double()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_unowned_outputs_fail_closed(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ("p07_01_step1." + fmt)).write_bytes(b"stale")
    with pytest.raises(TransactionError, match="unowned"):
        _run(source, cache, fmt, [_Pass()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_missing_committed_output_fails_closed(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_Pass()])
    (cache / ("p07_01_step1." + fmt)).unlink()
    with pytest.raises(TransactionError, match="missing"):
        _run(source, cache, fmt, [_Pass()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_transaction_option_off_keeps_legacy(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    p = _Pipeline(source, cache, fmt, [_Pass()])
    p.compile()
    p.forward(batch_size=2, resume_from_last=True)
    assert (cache / "p07_01_last_success_step.txt").exists()
    assert not (cache / "p07_01_stream_transaction.json").exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_force_killed_process_rolls_back_persisted_batch(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    # Exercise real production pipeline/storage; os._exit cannot run normal
    # exception handling, so recovery occurs in a fresh parent-side instance.
    script = textwrap.dedent("""
        import importlib.util
        import os
        import sys

        spec = importlib.util.spec_from_file_location("test_stream_transactional_resume", sys.argv[4])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        operator = mod._Pass()
        def crash_after_write(storage):
            data = storage.read("dataframe")
            storage.write(data)
            if data["id"].tolist() == [2, 3]:
                os._exit(79)
        operator.run = crash_after_write
        pipeline = mod._Pipeline(sys.argv[1], sys.argv[2], sys.argv[3], [operator])
        pipeline.compile()
        pipeline.forward(batch_size=2, transactional_resume=True)
    """)
    result = subprocess.run([sys.executable, "-c", script, str(source), str(cache), fmt, __file__],
                            capture_output=True, text=True)
    assert result.returncode == 79, result.stderr
    _run(source, cache, fmt, [_Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
@pytest.mark.parametrize("after_replace", [False, True])
def test_manifest_commit_uncertainty_reconciles(tmp_path, monkeypatch, fmt, after_replace):
    from dataflow.utils.stream_transaction import StreamTransactionCoordinator
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    operator = _Pass()
    p = _Pipeline(source, cache, fmt, [operator])
    p.compile()
    original = StreamTransactionCoordinator._atomic_manifest
    attempted = [False]

    def injected(self, state):
        if state["generation"] == 1 and not attempted[0]:
            attempted[0] = True
            if after_replace:
                original(self, state)
            raise OSError("injected manifest durability failure")
        return original(self, state)

    monkeypatch.setattr(StreamTransactionCoordinator, "_atomic_manifest", injected)
    with pytest.raises(TransactionError, match="commit uncertain"):
        p.forward(batch_size=2, transactional_resume=True)
    assert attempted[0]
    p.forward(batch_size=2, transactional_resume=True)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


class _DropFirst(OperatorABC):
    transaction_safe = True

    def transaction_signature(self):
        return {}

    def run(self, storage):
        frame = storage.read("dataframe")
        storage.write([] if frame["id"].tolist() == [0, 1] else frame)


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_empty_first_batch_then_populated(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_DropFirst()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [2, 3, 4]


@pytest.mark.cpu
def test_csv_embedded_newlines_are_logical_rows(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"id": [0, 1, 2, 3, 4],
                  "text": ["a\nb", "b", "c\n\nd", "d", "e"]}).to_csv(source, index=False)
    cache = tmp_path / "cache"
    _run(source, cache, "csv", [_Pass()])
    output = pd.read_csv(cache / "p07_01_step1.csv")
    assert output["id"].tolist() == [0, 1, 2, 3, 4]
    assert output["text"].tolist() == ["a\nb", "b", "c\n\nd", "d", "e"]


class _LifecyclePass(_Pass):
    def __init__(self, fail=None):
        super().__init__(fail=fail)
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeServing:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_stage_lifecycle_shutdown_and_shared_serving_cleanup(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    operators = [_LifecyclePass(), _LifecyclePass()]
    pipeline = _Pipeline(source, tmp_path / "cache", fmt, operators)
    pipeline.compile()
    serving = _FakeServing()
    pipeline.op_nodes_list[1].llm_serving = serving
    pipeline.op_nodes_list[2].llm_serving = serving
    pipeline.forward(batch_size=2, transactional_resume=True)
    assert [op.shutdown_calls for op in operators] == [1, 1]
    assert serving.cleanup_calls == 1
    assert pipeline.active_llm_serving is None
    pipeline.forward(batch_size=2, transactional_resume=True)
    assert [op.shutdown_calls for op in operators] == [1, 1]
    assert serving.cleanup_calls == 1


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_stage_failure_releases_serving_and_operator_resources(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    op = _LifecyclePass(fail="after")
    pipeline = _Pipeline(source, tmp_path / "cache", fmt, [op])
    pipeline.compile()
    serving = _FakeServing()
    pipeline.op_nodes_list[1].llm_serving = serving
    with pytest.raises(RuntimeError, match="after write"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    assert op.shutdown_calls == 1
    assert serving.cleanup_calls == 1
    assert pipeline.active_llm_serving is None
    pipeline.forward(batch_size=2, transactional_resume=True)
    assert op.shutdown_calls == 2
    assert serving.cleanup_calls == 2


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_competing_process_lock_and_release_after_forced_exit(tmp_path, fmt):
    from dataflow.utils.stream_transaction import StreamTransactionCoordinator
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    script = textwrap.dedent("""
        import importlib.util, sys
        from dataflow.utils.stream_transaction import StreamTransactionCoordinator
        spec = importlib.util.spec_from_file_location("test_stream_transactional_resume", sys.argv[4])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p = mod._Pipeline(sys.argv[1], sys.argv[2], sys.argv[3], [mod._Pass()])
        p.compile()
        with StreamTransactionCoordinator(p.op_nodes_list[1:-1], 2):
            from pathlib import Path
            Path(sys.argv[5]).write_text("acquired", encoding="utf-8")
            sys.stdin.readline()
    """)
    marker = tmp_path / "writer-ready"
    worker = subprocess.Popen(
        [sys.executable, "-c", script, str(source), str(cache), fmt, __file__, str(marker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # File-based readiness avoids blocking on stdout/stderr log buffers.
        import time
        deadline = time.monotonic() + 30
        while not marker.exists() and worker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists(), "Lock holder failed to signal readiness"
        # The second process must not reach operator.run or the manifest writer.
        with pytest.raises((TransactionError, OSError)):
            _run(source, cache, fmt, [_Pass()])
        assert worker.poll() is None
    finally:
        worker.kill()  # Simulates an abandoned OS lock on process exit.
        worker.communicate(timeout=15)
    _run(source, cache, fmt, [_Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
@pytest.mark.parametrize("damage", ["broken_json", "seal", "identity", "generation", "prefix"])
def test_malformed_manifest_fails_closed(tmp_path, fmt, damage):
    import hashlib
    import json
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_Pass()])
    path = cache / "p07_01_stream_transaction.json"
    state = json.loads(path.read_text())
    if damage == "broken_json":
        path.write_text("{broken", encoding="utf-8")
    else:
        if damage == "seal":
            state["seal"] = "0" * 64
        else:
            if damage == "identity":
                state["identity"] = "wrong"
            elif damage == "generation":
                state["generation"] = -1
            elif damage == "prefix":
                state["outputs"][0]["sha256"] = "0" * 64
            payload = {k: v for k, v in state.items() if k != "seal"}
            state["seal"] = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False
            ).encode("utf-8")).hexdigest()
        path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(TransactionError):
        _run(source, cache, fmt, [_Pass()])


@pytest.mark.cpu
def test_opt_in_operator_contract_rejects_unknown_operator(tmp_path):
    source = _source(tmp_path, "jsonl")
    op = _Pass()
    op.transaction_safe = False
    p = _Pipeline(source, tmp_path / "cache", "jsonl", [op])
    p.compile()
    with pytest.raises(TransactionError, match="transaction_safe"):
        p.forward(batch_size=2, transactional_resume=True)


class _AsyncCleanupServing:
    def __init__(self):
        self.cleanup_calls = 0

    async def cleanup(self):
        self.cleanup_calls += 1


class _ShutdownFailsAfterCommit(_Pass):
    def shutdown(self):
        raise RuntimeError("injected shutdown failure")


class _AsyncShutdownOperator(_Pass):
    async def shutdown(self):
        return None


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_async_serving_cleanup_is_rejected_before_execution(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    pipeline = _Pipeline(source, cache, fmt, [_Pass()])
    pipeline.compile()
    serving = _AsyncCleanupServing()
    pipeline.op_nodes_list[1].llm_serving = serving
    with pytest.raises(TransactionError, match="Async serving cleanup"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    assert serving.cleanup_calls == 0
    assert not (cache / ("p07_01_step1." + fmt)).exists()
    assert not (cache / "p07_01_stream_transaction.json").exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_async_operator_shutdown_is_rejected_before_execution(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    pipeline = _Pipeline(source, cache, fmt, [_AsyncShutdownOperator()])
    pipeline.compile()
    with pytest.raises(TransactionError, match="Async operator shutdown"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    assert not (cache / ("p07_01_step1." + fmt)).exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_noncontiguous_serving_reuse_is_rejected_before_execution(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    pipeline = _Pipeline(source, cache, fmt, [_Pass(), _Pass(), _Pass()])
    pipeline.compile()
    serving_a = _FakeServing()
    serving_b = _FakeServing()
    for node, serving in zip(pipeline.op_nodes_list[1:4],
                             [serving_a, serving_b, serving_a]):
        node.llm_serving = serving
    with pytest.raises(TransactionError, match="Non-contiguous serving reuse"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    assert serving_a.cleanup_calls == serving_b.cleanup_calls == 0
    assert not (cache / ("p07_01_step1." + fmt)).exists()
    assert not (cache / "p07_01_stream_transaction.json").exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_shutdown_failure_after_stage_commit_requires_process_restart(tmp_path, fmt):
    import json
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    pipeline = _Pipeline(source, cache, fmt,
                         [_ShutdownFailsAfterCommit(), _Pass()])
    pipeline.compile()
    with pytest.raises(TransactionError, match="shutdown failed after durable stage transition"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    checkpoint = json.loads((cache / "p07_01_stream_transaction.json").read_text())
    assert (checkpoint["stage"], checkpoint["batch"]) == (1, 0)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]
    assert not (cache / ("p07_01_step2." + fmt)).exists()
    # Neither the same instance nor another instance in this process can
    # silently continue with resources whose cleanup failed.
    with pytest.raises(TransactionError, match="fresh process"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    with pytest.raises(TransactionError, match="restart the process"):
        _run(source, cache, fmt, [_ShutdownFailsAfterCommit(), _Pass()])

    # A genuinely new interpreter process can resume at committed stage 1.
    script = textwrap.dedent("""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "test_stream_transactional_resume", sys.argv[4]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipeline = module._Pipeline(
            sys.argv[1], sys.argv[2], sys.argv[3],
            [module._ShutdownFailsAfterCommit(), module._Pass()]
        )
        pipeline.compile()
        pipeline.forward(batch_size=2, transactional_resume=True)
    """)
    child = subprocess.run(
        [sys.executable, "-c", script, str(source), str(cache), fmt, __file__],
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]
    assert _output(cache / ("p07_01_step2." + fmt), fmt) == [0, 1, 2, 3, 4]


class _SyncShutdownReturnsCoroutine(_Pass):
    """The shutdown method itself is sync, but its return value is awaitable."""

    def __init__(self):
        super().__init__()
        self.shutdown_calls = 0
        self.async_cleanup_ran = False

    def shutdown(self):
        self.shutdown_calls += 1

        async def actually_cleanup():
            self.async_cleanup_ran = True

        return actually_cleanup()


class _AbortShutdownRaises(_Pass):
    """Operator.run raises first; shutdown raises while handling that error."""

    def __init__(self):
        super().__init__(fail="after")
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        raise RuntimeError("secondary shutdown failure")


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_sync_shutdown_returning_awaitable_fails_closed_after_durable_transition(tmp_path, fmt):
    import json

    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    operator = _SyncShutdownReturnsCoroutine()
    pipeline = _Pipeline(source, cache, fmt, [operator, _Pass()])
    pipeline.compile()
    with pytest.raises(TransactionError, match="shutdown returned an awaitable"):
        pipeline.forward(batch_size=2, transactional_resume=True)

    assert operator.shutdown_calls == 1
    assert operator.async_cleanup_ran is False
    checkpoint = json.loads((cache / "p07_01_stream_transaction.json").read_text())
    assert (checkpoint["stage"], checkpoint["batch"]) == (1, 0)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]
    assert not (cache / ("p07_01_step2." + fmt)).exists()
    with pytest.raises(TransactionError, match="fresh process"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    with pytest.raises(TransactionError, match="restart the process"):
        _run(source, cache, fmt, [_SyncShutdownReturnsCoroutine(), _Pass()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_abort_shutdown_failure_preserves_primary_error_and_blocks_retry(tmp_path, fmt):
    import json

    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    operator = _AbortShutdownRaises()
    pipeline = _Pipeline(source, cache, fmt, [operator])
    pipeline.compile()
    with pytest.raises(RuntimeError, match="after write") as primary:
        pipeline.forward(batch_size=2, transactional_resume=True)

    assert "secondary shutdown failure" not in str(primary.value)
    assert operator.shutdown_calls == 1
    checkpoint = json.loads((cache / "p07_01_stream_transaction.json").read_text())
    assert (checkpoint["stage"], checkpoint["batch"]) == (0, 1)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1]
    with pytest.raises(TransactionError, match="fresh process"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    with pytest.raises(TransactionError, match="restart the process"):
        _run(source, cache, fmt, [_AbortShutdownRaises()])
