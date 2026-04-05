"""CPU-only pytest suite: RayAcceleratedOperator with PipelineABC.compile().

Verifies that RayAcceleratedOperator works correctly through compile() for
all three pipeline types, with multi-operator chains, deterministic content,
and preserved row ordering.

Run:
    pytest test/rayorch/test_compile_cpu.py -v -m cpu
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from dataflow.pipeline.Pipeline import (
    BatchedPipelineABC,
    PipelineABC,
    StreamBatchedPipelineABC,
)
from dataflow.rayorch import RayAcceleratedOperator
from dataflow.rayorch._test_ops import DummyDoubleOp, DummyIncrementOp

N_ROWS = 100
REPLICAS = 4


# =====================================================================
# Pipeline definitions for each ABC
# =====================================================================
def _make_storage(cls, input_file: str, cache_path: str):
    """Instantiate the correct FileStorage subclass."""
    return cls(
        first_entry_file_name=input_file,
        cache_path=cache_path,
        file_name_prefix="step",
        cache_type="jsonl",
    )


class _SerialPipeline(PipelineABC):
    def __init__(self, input_file, cache_path, storage_cls):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = DummyDoubleOp()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _RayPipeline(PipelineABC):
    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _SerialBatched(BatchedPipelineABC):
    def __init__(self, input_file, cache_path, storage_cls):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = DummyDoubleOp()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _RayBatched(BatchedPipelineABC):
    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _SerialStreamBatched(StreamBatchedPipelineABC):
    def __init__(self, input_file, cache_path, storage_cls):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = DummyDoubleOp()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _RayStreamBatched(StreamBatchedPipelineABC):
    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = DummyIncrementOp()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def ray_env():
    import ray

    ray.init(ignore_reinit_error=True, num_cpus=8)
    yield
    ray.shutdown()


@pytest.fixture()
def test_data(tmp_path):
    """Create a deterministic input JSONL with a 'value' column [0..N_ROWS)."""
    df = pd.DataFrame({"value": list(range(N_ROWS))})
    input_file = str(tmp_path / "input.jsonl")
    df.to_json(input_file, orient="records", lines=True, force_ascii=False)
    return input_file, tmp_path


# =====================================================================
# Helpers
# =====================================================================
def _read_final_output(cache_path: str) -> pd.DataFrame:
    """Read the step-2 output produced by a two-operator pipeline."""
    return pd.read_json(
        os.path.join(cache_path, "step_step2.jsonl"), lines=True,
    )


def _shutdown_ray_actors(pipe):
    for op_node in getattr(pipe, "op_nodes_list", []):
        if hasattr(op_node.op_obj, "shutdown"):
            op_node.op_obj.shutdown()


def _assert_content(df: pd.DataFrame):
    """Verify deterministic content: doubled == value*2, incremented == doubled+1."""
    np.testing.assert_array_equal(df["doubled"].values, df["value"].values * 2)
    np.testing.assert_array_equal(df["incremented"].values, df["doubled"].values + 1)


def _assert_ordering(serial_df: pd.DataFrame, ray_df: pd.DataFrame):
    """Verify row ordering: Ray output must match serial output exactly."""
    np.testing.assert_array_equal(ray_df["value"].values, serial_df["value"].values)
    np.testing.assert_array_equal(ray_df["doubled"].values, serial_df["doubled"].values)
    np.testing.assert_array_equal(
        ray_df["incremented"].values, serial_df["incremented"].values,
    )


# =====================================================================
# Tests — PipelineABC
# =====================================================================
@pytest.mark.cpu
def test_pipeline_abc(ray_env, test_data):
    """PipelineABC: compile → forward with Ray(DummyDouble) → DummyIncrement."""
    from dataflow.utils.storage import FileStorage

    input_file, tmp_path = test_data

    serial_cache = str(tmp_path / "pipe_serial")
    os.makedirs(serial_cache, exist_ok=True)
    serial_pipe = _SerialPipeline(input_file, serial_cache, FileStorage)
    serial_pipe.compile()
    serial_pipe.forward()
    serial_df = _read_final_output(serial_cache)

    ray_cache = str(tmp_path / "pipe_ray")
    os.makedirs(ray_cache, exist_ok=True)
    ray_pipe = _RayPipeline(input_file, ray_cache, FileStorage)
    ray_pipe.compile()
    ray_pipe.forward()
    ray_df = _read_final_output(ray_cache)

    _assert_content(ray_df)
    _assert_ordering(serial_df, ray_df)
    _shutdown_ray_actors(ray_pipe)


# =====================================================================
# Tests — BatchedPipelineABC (no batching)
# =====================================================================
@pytest.mark.cpu
def test_batched_pipeline_abc(ray_env, test_data):
    """BatchedPipelineABC: compile → forward (batch_size=None)."""
    from dataflow.utils.storage import BatchedFileStorage

    input_file, tmp_path = test_data

    serial_cache = str(tmp_path / "batched_serial")
    os.makedirs(serial_cache, exist_ok=True)
    serial_pipe = _SerialBatched(input_file, serial_cache, BatchedFileStorage)
    serial_pipe.compile()
    serial_pipe.forward(resume_from_last=False)
    serial_df = _read_final_output(serial_cache)

    ray_cache = str(tmp_path / "batched_ray")
    os.makedirs(ray_cache, exist_ok=True)
    ray_pipe = _RayBatched(input_file, ray_cache, BatchedFileStorage)
    ray_pipe.compile()
    ray_pipe.forward(resume_from_last=False)
    ray_df = _read_final_output(ray_cache)

    _assert_content(ray_df)
    _assert_ordering(serial_df, ray_df)
    _shutdown_ray_actors(ray_pipe)


# =====================================================================
# Tests — BatchedPipelineABC WITH batch_size
# =====================================================================
@pytest.mark.cpu
def test_batched_pipeline_abc_with_batch_size(ray_env, test_data):
    """BatchedPipelineABC: compile → forward(batch_size=30).

    RayAcceleratedOperator is called multiple times (once per batch),
    and the output must still be content-correct and order-preserving.
    """
    from dataflow.utils.storage import BatchedFileStorage

    input_file, tmp_path = test_data
    batch_size = 30

    serial_cache = str(tmp_path / "batched_bs_serial")
    os.makedirs(serial_cache, exist_ok=True)
    serial_pipe = _SerialBatched(input_file, serial_cache, BatchedFileStorage)
    serial_pipe.compile()
    serial_pipe.forward(batch_size=batch_size, resume_from_last=False)
    serial_df = _read_final_output(serial_cache)

    ray_cache = str(tmp_path / "batched_bs_ray")
    os.makedirs(ray_cache, exist_ok=True)
    ray_pipe = _RayBatched(input_file, ray_cache, BatchedFileStorage)
    ray_pipe.compile()
    ray_pipe.forward(batch_size=batch_size, resume_from_last=False)
    ray_df = _read_final_output(ray_cache)

    assert len(ray_df) == N_ROWS, f"Expected {N_ROWS} rows, got {len(ray_df)}"
    _assert_content(ray_df)
    _assert_ordering(serial_df, ray_df)
    _shutdown_ray_actors(ray_pipe)


# =====================================================================
# Tests — StreamBatchedPipelineABC (no batching)
# =====================================================================
@pytest.mark.cpu
def test_stream_batched_pipeline_abc(ray_env, test_data):
    """StreamBatchedPipelineABC: compile → forward (batch_size=None)."""
    from dataflow.utils.storage import StreamBatchedFileStorage

    input_file, tmp_path = test_data

    serial_cache = str(tmp_path / "stream_serial")
    os.makedirs(serial_cache, exist_ok=True)
    serial_pipe = _SerialStreamBatched(input_file, serial_cache, StreamBatchedFileStorage)
    serial_pipe.compile()
    serial_pipe.forward(resume_from_last=False)
    serial_df = _read_final_output(serial_cache)

    ray_cache = str(tmp_path / "stream_ray")
    os.makedirs(ray_cache, exist_ok=True)
    ray_pipe = _RayStreamBatched(input_file, ray_cache, StreamBatchedFileStorage)
    ray_pipe.compile()
    ray_pipe.forward(resume_from_last=False)
    ray_df = _read_final_output(ray_cache)

    _assert_content(ray_df)
    _assert_ordering(serial_df, ray_df)
    _shutdown_ray_actors(ray_pipe)


# =====================================================================
# Tests — ordering with more replicas
# =====================================================================
@pytest.mark.cpu
def test_ordering_many_replicas(ray_env, test_data):
    """Verify SHARD_CONTIGUOUS preserves order even with many replicas."""
    from dataflow.utils.storage import FileStorage

    input_file, tmp_path = test_data

    for replicas in (2, 3, 7):
        cache = str(tmp_path / f"order_{replicas}r")
        os.makedirs(cache, exist_ok=True)
        pipe = _RayPipeline(input_file, cache, FileStorage, replicas=replicas)
        pipe.compile()
        pipe.forward()
        df = _read_final_output(cache)

        expected_values = list(range(N_ROWS))
        assert list(df["value"]) == expected_values, (
            f"Ordering broken with {replicas} replicas"
        )
        _assert_content(df)
        _shutdown_ray_actors(pipe)


# =====================================================================
# Pipeline: both operators are RayAcceleratedOperator (for auto-shutdown test)
# =====================================================================
class _AllRayPipeline(PipelineABC):
    """Both stages are RayAcceleratedOperators."""

    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = RayAcceleratedOperator(
            DummyIncrementOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _AllRayBatched(BatchedPipelineABC):
    """Both stages are RayAcceleratedOperators (batched)."""

    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = RayAcceleratedOperator(
            DummyIncrementOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


class _AllRayStreamBatched(StreamBatchedPipelineABC):
    """Both stages are RayAcceleratedOperators (stream batched)."""

    def __init__(self, input_file, cache_path, storage_cls, replicas=REPLICAS):
        super().__init__()
        self.storage = _make_storage(storage_cls, input_file, cache_path)
        self.doubler = RayAcceleratedOperator(
            DummyDoubleOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()
        self.incrementer = RayAcceleratedOperator(
            DummyIncrementOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


# =====================================================================
# Tests — serial-then-ray ordering
# =====================================================================
class _SerialThenRayPipeline(PipelineABC):
    """Chain: serial DummyDouble → Ray(DummyIncrement)."""

    def __init__(self, input_file, cache_path, replicas=REPLICAS):
        super().__init__()
        from dataflow.utils.storage import FileStorage

        self.storage = FileStorage(input_file, cache_path, "step", "jsonl")
        self.doubler = DummyDoubleOp()
        self.incrementer = RayAcceleratedOperator(
            DummyIncrementOp, replicas=replicas, num_gpus_per_replica=0.0,
        ).op_cls_init()

    def forward(self):
        self.doubler.run(
            storage=self.storage.step(),
            input_key="value",
            output_key="doubled",
        )
        self.incrementer.run(
            storage=self.storage.step(),
            input_key="doubled",
            output_key="incremented",
        )


@pytest.mark.cpu
def test_serial_then_ray(ray_env, test_data):
    """DummyDouble(serial) → DummyIncrement(Ray): verify content and order."""
    from dataflow.utils.storage import FileStorage

    input_file, tmp_path = test_data

    serial_cache = str(tmp_path / "sr_serial")
    os.makedirs(serial_cache, exist_ok=True)
    serial_pipe = _SerialPipeline(input_file, serial_cache, FileStorage)
    serial_pipe.compile()
    serial_pipe.forward()
    serial_df = _read_final_output(serial_cache)

    ray_cache = str(tmp_path / "sr_ray")
    os.makedirs(ray_cache, exist_ok=True)
    ray_pipe = _SerialThenRayPipeline(input_file, ray_cache)
    ray_pipe.compile()
    ray_pipe.forward()
    ray_df = _read_final_output(ray_cache)

    _assert_content(ray_df)
    _assert_ordering(serial_df, ray_df)
    _shutdown_ray_actors(ray_pipe)


# =====================================================================
# Tests — auto-shutdown of RayAcceleratedOperator in _compiled_forward
# =====================================================================
def _get_ray_op_nodes(pipe) -> list:
    """Return OperatorNode entries whose op_obj has a shutdown method
    (i.e. is a RayAcceleratedOperator)."""
    return [
        n for n in getattr(pipe, "op_nodes_list", [])
        if hasattr(getattr(n, "op_obj", None), "shutdown")
    ]


@pytest.mark.cpu
def test_auto_shutdown_pipeline_abc(ray_env, test_data):
    """PipelineABC with two RayAcceleratedOperators: verify actors are
    automatically shut down after each stage and output is still correct."""
    from dataflow.utils.storage import FileStorage

    input_file, tmp_path = test_data

    cache = str(tmp_path / "auto_sd_pipe")
    os.makedirs(cache, exist_ok=True)
    pipe = _AllRayPipeline(input_file, cache, FileStorage)
    pipe.compile()
    pipe.forward()

    df = _read_final_output(cache)
    _assert_content(df)
    assert len(df) == N_ROWS

    for node in _get_ray_op_nodes(pipe):
        assert node.op_obj._module is None, (
            f"{node.op_name} was not auto-shutdown after compiled forward"
        )


@pytest.mark.cpu
def test_auto_shutdown_batched(ray_env, test_data):
    """BatchedPipelineABC with two Ray stages: auto-shutdown after batches."""
    from dataflow.utils.storage import BatchedFileStorage

    input_file, tmp_path = test_data

    cache = str(tmp_path / "auto_sd_batched")
    os.makedirs(cache, exist_ok=True)
    pipe = _AllRayBatched(input_file, cache, BatchedFileStorage)
    pipe.compile()
    pipe.forward(batch_size=30, resume_from_last=False)

    df = _read_final_output(cache)
    _assert_content(df)
    assert len(df) == N_ROWS

    for node in _get_ray_op_nodes(pipe):
        assert node.op_obj._module is None, (
            f"{node.op_name} was not auto-shutdown after compiled forward"
        )


@pytest.mark.cpu
def test_auto_shutdown_stream_batched(ray_env, test_data):
    """StreamBatchedPipelineABC with two Ray stages: auto-shutdown after stream batches."""
    from dataflow.utils.storage import StreamBatchedFileStorage

    input_file, tmp_path = test_data

    cache = str(tmp_path / "auto_sd_stream")
    os.makedirs(cache, exist_ok=True)
    pipe = _AllRayStreamBatched(input_file, cache, StreamBatchedFileStorage)
    pipe.compile()
    pipe.forward(resume_from_last=False)

    df = _read_final_output(cache)
    _assert_content(df)
    assert len(df) == N_ROWS

    for node in _get_ray_op_nodes(pipe):
        assert node.op_obj._module is None, (
            f"{node.op_name} was not auto-shutdown after compiled forward"
        )
