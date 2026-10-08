"""Core text operators must answer every input row, even when some cells are empty."""

import asyncio

import pandas as pd
import pytest

from dataflow.operators.core_text import (
    PromptedEvaluator,
    PromptedFilter,
    PromptedGenerator,
    PromptedRefiner,
)

pytestmark = pytest.mark.cpu

ROWS = ["A", "B", "C"]


class MemoryStorage:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.dataframe

    def write(self, dataframe):
        self.result = dataframe
        return "memory://core_text"


def _label_responses(user_inputs):
    return [f"response {position}" for position in range(len(user_inputs))]


def _score_responses(user_inputs):
    return [str(5 - position) for position in range(len(user_inputs))]


class StubServing:
    """Answers one response per prompt and records the batches it received."""

    def __init__(self, responder=_label_responses):
        self.requests = []
        self.responder = responder

    def batches(self):
        return [len(batch) for batch in self.requests]

    def generate_from_input(self, user_inputs=None, system_prompt=None, json_schema=None, **kwargs):
        self.requests.append(list(user_inputs))
        return self.responder(self.requests[-1])


class AsyncStubServing(StubServing):
    async def generate_from_input(self, user_inputs=None, system_prompt=None, json_schema=None, **kwargs):
        return StubServing.generate_from_input(self, user_inputs, system_prompt, json_schema, **kwargs)


def source_frame(empty_position=None):
    """Three rows with deliberately non-contiguous labels."""
    frame = pd.DataFrame(
        {
            "id": ROWS,
            "raw_content": ["first", "second", "third"],
        },
        index=[3, 7, 11],
    )
    if empty_position is not None:
        frame.loc[frame.index[empty_position], "raw_content"] = ""
    return frame


def expected_labels(empty_position):
    positions = [position for position in range(len(ROWS)) if position != empty_position]
    labels = {position: "" for position in range(len(ROWS))}
    for slot, position in enumerate(positions):
        labels[position] = f"response {slot}"
    return [labels[position] for position in range(len(ROWS))]


@pytest.mark.parametrize("empty_position", [0, 1, 2, None])
def test_prompted_generator_keeps_row_alignment(empty_position):
    serving = StubServing()
    storage = MemoryStorage(source_frame(empty_position))
    operator = PromptedGenerator(llm_serving=serving, user_prompt="Generate: ")

    output_key = operator.run(storage=storage, input_key="raw_content", output_key="generated")

    assert output_key == "generated"
    assert list(storage.result["id"]) == ROWS
    assert list(storage.result["generated"]) == expected_labels(empty_position)
    assert serving.batches() == [3 if empty_position is None else 2]


@pytest.mark.parametrize("empty_position", [0, 1, 2, None])
def test_retrieval_generator_keeps_row_alignment(empty_position):
    # RetrievalGenerator imports dataflow.serving eagerly, so it needs the optional cloud extras.
    module = pytest.importorskip("dataflow.operators.core_text.generate.retrieval_generator")
    serving = AsyncStubServing()
    storage = MemoryStorage(source_frame(empty_position))
    operator = module.RetrievalGenerator(llm_serving=serving)

    output_key = asyncio.run(
        operator.run(storage=storage, input_key="raw_content", output_key="generated")
    )

    assert output_key == "generated"
    assert list(storage.result["id"]) == ROWS
    assert list(storage.result["generated"]) == expected_labels(empty_position)


@pytest.mark.parametrize("empty_position", [0, 1, 2, None])
def test_prompted_refiner_keeps_row_alignment(empty_position):
    serving = StubServing()
    storage = MemoryStorage(source_frame(empty_position))
    operator = PromptedRefiner(llm_serving=serving)

    operator.run(storage=storage, input_key="raw_content")

    refined = list(storage.result["raw_content"])
    assert list(storage.result["id"]) == ROWS
    if empty_position is None:
        assert refined == expected_labels(None)
    else:
        assert refined[empty_position] == ""
        assert [value for position, value in enumerate(refined) if position != empty_position] == [
            "response 0",
            "response 1",
        ]


@pytest.mark.parametrize("empty_position", [0, 1, 2, None])
def test_prompted_evaluator_scores_every_row(empty_position):
    serving = StubServing(responder=_score_responses)
    storage = MemoryStorage(source_frame(empty_position))
    operator = PromptedEvaluator(llm_serving=serving)

    output_key = operator.run(storage=storage, input_key="raw_content", output_key="eval")

    assert output_key == "eval"
    assert list(storage.result["id"]) == ROWS
    scores = list(storage.result["eval"])
    assert len(scores) == len(ROWS)
    expected = [5 - slot for slot in range(2 if empty_position is not None else 3)]
    if empty_position is None:
        assert scores == expected
    else:
        assert scores[empty_position] == 0
        assert [value for position, value in enumerate(scores) if position != empty_position] == expected


def test_prompted_filter_still_drops_empty_rows():
    serving = StubServing(responder=_score_responses)
    storage = MemoryStorage(source_frame(empty_position=1))
    operator = PromptedFilter(llm_serving=serving, min_score=1, max_score=5)

    output_key = operator.run(storage=storage, input_key="raw_content", output_key="eval")

    assert output_key == "eval"
    assert list(storage.result["id"]) == ["A", "C"]
    assert list(storage.result["eval"]) == [5, 4]


def test_short_response_batch_is_reported_instead_of_shifted():
    serving = StubServing(responder=lambda user_inputs: ["response 0"])
    storage = MemoryStorage(source_frame())
    operator = PromptedGenerator(llm_serving=serving)

    with pytest.raises(RuntimeError, match="Response count mismatch"):
        operator.run(storage=storage, input_key="raw_content", output_key="generated")
