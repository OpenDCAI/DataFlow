"""CPU-only regression tests for PipelineABC compilation lifecycle (P11.01/P11.02)."""

from collections import Counter

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline.Pipeline import (
    BatchedPipelineABC,
    PipelineABC,
    StreamBatchedPipelineABC,
)
from dataflow.utils.storage import DataFlowStorage
from dataflow.wrapper.auto_op import AutoOP


class _MemoryStorage(DataFlowStorage):
    """Exercise pipeline compilation without files, Ray, or model services."""

    def __init__(self):
        self.operator_step = 0
        self.dataframe = pd.DataFrame({"value": [1, 2, 3]})

    def step(self):
        self.operator_step += 1
        return self

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, data):
        self.dataframe = data.copy()


class _Increment(OperatorABC):
    def __init__(self):
        super().__init__()
        self.run_calls = 0

    def run(self, storage, input_key, output_key):
        self.run_calls += 1
        data = storage.read()
        data[output_key] = data[input_key] + 1
        storage.write(data)


def _make_pipeline(base, failure=None):
    class ExamplePipeline(base):
        def __init__(self):
            super().__init__()
            self.storage = _MemoryStorage()
            self.first = _Increment()
            self.second = _Increment()
            self.failure = failure

        def forward(self):
            self.first.run(
                storage=self.storage.step(),
                input_key="value",
                output_key="middle",
            )
            if self.failure == "forward":
                raise RuntimeError("intentional compile-time forward failure")

            second_input = "missing" if self.failure == "graph" else "middle"
            if self.failure == "storage":
                self.second.run(input_key=second_input, output_key="result")
            else:
                self.second.run(
                    storage=self.storage.step(),
                    input_key=second_input,
                    output_key="result",
                )

    return ExamplePipeline()


@pytest.mark.cpu
@pytest.mark.parametrize("base", [PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC])
@pytest.mark.parametrize(
    ("failure", "exception"),
    [
        ("forward", RuntimeError),
        ("graph", KeyError),
        ("storage", TypeError),
    ],
)
def test_failed_compile_rolls_back_and_retry_succeeds(base, failure, exception):
    pipeline = _make_pipeline(base, failure=failure)
    first, second = pipeline.first, pipeline.second
    original_forward = pipeline.forward.__func__

    with pytest.raises(exception):
        pipeline.compile()

    assert pipeline.compiled is False
    assert pipeline.forward.__func__ is original_forward
    assert pipeline.first is first
    assert pipeline.second is second
    assert pipeline.storage.operator_step == 0
    assert pipeline.op_runtimes == []
    assert pipeline.op_nodes_list == []
    assert pipeline.accumulated_keys == []
    assert pipeline.llm_serving_list == []
    assert pipeline.llm_serving_counter == Counter()
    assert not hasattr(pipeline, "final_keys")
    assert not hasattr(pipeline, "input_dataset_node")
    assert not hasattr(pipeline, "output_dataset_node")
    assert not hasattr(pipeline, "last_modified_index_of_keys")
    assert first.run_calls == second.run_calls == 0

    pipeline.failure = None
    pipeline.compile()
    assert pipeline.compiled is True
    assert len(pipeline.op_runtimes) == 2
    assert len(pipeline.op_nodes_list) == 4
    assert pipeline.storage.operator_step == 2
    assert pipeline.op_runtimes[0].kwargs["storage"] is pipeline.storage

    if base is PipelineABC:
        pipeline.forward()
    else:
        pipeline.forward(resume_from_last=False)
    assert pipeline.storage.dataframe["result"].tolist() == [3, 4, 5]
    assert (first.run_calls, second.run_calls) == (1, 1)


@pytest.mark.cpu
@pytest.mark.parametrize("base", [PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC])
def test_repeated_compile_does_not_reregister_or_execute(base):
    pipeline = _make_pipeline(base)
    pipeline.compile()
    state = (
        tuple(map(id, pipeline.op_runtimes)),
        tuple(map(id, pipeline.op_nodes_list)),
        tuple(tuple(keys) for keys in pipeline.accumulated_keys),
        pipeline.storage.operator_step,
        dict(pipeline.llm_serving_counter),
    )
    assert isinstance(pipeline.first, AutoOP)
    assert isinstance(pipeline.second, AutoOP)
    assert pipeline.first._operator.run_calls == 0
    assert pipeline.second._operator.run_calls == 0

    pipeline.compile()
    pipeline.compile()

    assert state == (
        tuple(map(id, pipeline.op_runtimes)),
        tuple(map(id, pipeline.op_nodes_list)),
        tuple(tuple(keys) for keys in pipeline.accumulated_keys),
        pipeline.storage.operator_step,
        dict(pipeline.llm_serving_counter),
    )
    assert pipeline.first._operator.run_calls == 0
    assert pipeline.second._operator.run_calls == 0
    assert len(pipeline.op_runtimes) == 2
    assert len(pipeline.op_nodes_list) == 4

    if base is PipelineABC:
        pipeline.forward()
    else:
        pipeline.forward(resume_from_last=False)
    assert pipeline.storage.dataframe["result"].tolist() == [3, 4, 5]
    assert pipeline.first._operator.run_calls == 1
    assert pipeline.second._operator.run_calls == 1


class _StorageHolder:
    """A common user-owned wrapper not directly stored as pipeline.storage."""

    def __init__(self, storage):
        self.storage = storage


@pytest.mark.cpu
@pytest.mark.parametrize("base", [PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC])
@pytest.mark.parametrize("failure_stage", ["forward", "after_graph"])
def test_indirect_storage_recovers_after_failed_compile(base, failure_stage):
    class NestedStoragePipeline(base):
        def __init__(self):
            super().__init__()
            self.settings = {"nested": [_StorageHolder(_MemoryStorage())]}
            self.operator = _Increment()
            self.failure_stage = failure_stage

        def forward(self):
            self.operator.run(
                storage=self.settings["nested"][0].storage.step(),
                input_key="value",
                output_key="result",
            )
            if self.failure_stage == "forward":
                raise RuntimeError("failure after nested storage advancement")

        def _build_operator_nodes_graph(self):
            super()._build_operator_nodes_graph()
            if self.failure_stage == "after_graph":
                raise RuntimeError("failure after complete graph construction")

    pipeline = NestedStoragePipeline()
    storage = pipeline.settings["nested"][0].storage
    original_operator = pipeline.operator
    original_forward = pipeline.forward.__func__

    with pytest.raises(RuntimeError):
        pipeline.compile()

    assert storage.operator_step == 0
    assert pipeline.compiled is False
    assert pipeline.forward.__func__ is original_forward
    assert pipeline.operator is original_operator
    assert pipeline.op_runtimes == []
    assert pipeline.op_nodes_list == []
    assert pipeline.accumulated_keys == []
    assert not hasattr(pipeline, "input_dataset_node")
    assert not hasattr(pipeline, "output_dataset_node")
    assert not hasattr(pipeline, "final_keys")
    assert original_operator.run_calls == 0

    pipeline.failure_stage = None
    pipeline.compile()
    pipeline.compile()  # Repeated compilation must still be a no-op.
    assert storage.operator_step == 1
    assert len(pipeline.op_runtimes) == 1
    assert pipeline.op_runtimes[0].kwargs["storage"] is storage
    assert len(pipeline.op_nodes_list) == 3
    if base is PipelineABC:
        pipeline.forward()
    else:
        pipeline.forward(resume_from_last=False)
    assert storage.dataframe["result"].tolist() == [2, 3, 4]
    assert original_operator.run_calls == 1
