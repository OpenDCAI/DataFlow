"""CPU-only regression tests for PipelineABC compilation lifecycle (P11.01/P11.02)."""

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
def test_failed_compile_marks_pipeline_unusable(base, failure, exception):
    pipeline = _make_pipeline(base, failure=failure)
    original_forward = pipeline.forward.__func__

    with pytest.raises(exception):
        pipeline.compile()

    assert pipeline.compiled is False
    assert pipeline.forward.__func__ is original_forward

    pipeline.failure = None
    with pytest.raises(RuntimeError, match="create a new pipeline instance"):
        pipeline.compile()
    assert pipeline.compiled is False

    fresh = _make_pipeline(base, failure=None)
    fresh.compile()
    assert fresh.compiled is True
    assert len(fresh.op_runtimes) == 2
    assert len(fresh.op_nodes_list) == 4
    assert fresh.storage.operator_step == 2
    assert fresh.op_runtimes[0].kwargs["storage"] is fresh.storage

    if base is PipelineABC:
        fresh.forward()
    else:
        fresh.forward(resume_from_last=False)
    assert fresh.storage.dataframe["result"].tolist() == [3, 4, 5]
    assert (fresh.first._operator.run_calls, fresh.second._operator.run_calls) == (1, 1)


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


@pytest.mark.cpu
@pytest.mark.parametrize("base", [PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC])
def test_after_graph_compile_failure_is_one_shot(base):
    class AfterGraphFailPipeline(base):
        def __init__(self):
            super().__init__()
            self.storage = _MemoryStorage()
            self.operator = _Increment()

        def forward(self):
            self.operator.run(
                storage=self.storage.step(),
                input_key="value",
                output_key="result",
            )

        def _build_operator_nodes_graph(self):
            super()._build_operator_nodes_graph()
            raise RuntimeError("failure after complete graph construction")

    pipeline = AfterGraphFailPipeline()
    original_forward = pipeline.forward.__func__

    with pytest.raises(RuntimeError, match="failure after complete graph construction"):
        pipeline.compile()

    assert pipeline.compiled is False
    assert pipeline.forward.__func__ is original_forward
    with pytest.raises(RuntimeError, match="create a new pipeline instance"):
        pipeline.compile()
