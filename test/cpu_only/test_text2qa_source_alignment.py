"""Text2QA must keep generated prompts attached to their source documents."""

import json

import pandas as pd
import pytest

from dataflow.operators.core_text import Text2QAGenerator

pytestmark = pytest.mark.cpu


class MemoryStorage:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.dataframe

    def write(self, dataframe):
        self.result = dataframe
        return "memory://text2qa"


class StubServing:
    def __init__(self, prompt_responses):
        self.prompt_responses = prompt_responses
        self.calls = []

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls.append(list(user_inputs))
        if len(self.calls) == 1:
            return self.prompt_responses
        return [f"Q: Question {i}\nA: Answer {i}" for i in range(len(user_inputs))]


def source_frame():
    return pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "text": ["Source A", "Source B", "Source C"],
            "metadata": [10, 20, 30],
        },
        index=[4, 8, 15],
    )


@pytest.mark.parametrize("invalid_index", [0, 1, 2, None])
def test_prompt_expansion_preserves_source_rows(invalid_index):
    source = source_frame()
    replies = [
        json.dumps([f"Prompt {row_id}.1", f"Prompt {row_id}.2"])
        for row_id in source["id"]
    ]
    if invalid_index is not None:
        replies[invalid_index] = "invalid JSON"
    serving = StubServing(replies)
    storage = MemoryStorage(source)

    result_keys = Text2QAGenerator(serving).run(storage, input_question_num=2)

    valid_positions = [i for i in range(len(source)) if i != invalid_index]
    expected_positions = [i for i in valid_positions for _ in range(2)]
    expected_source = source.iloc[expected_positions].reset_index(drop=True)
    pd.testing.assert_frame_equal(storage.result[source.columns], expected_source)
    assert storage.result["generated_prompt"].tolist() == [
        f"Prompt {source.iloc[i]['id']}.{j}" for i in valid_positions for j in (1, 2)
    ]
    for generated_input, row in zip(serving.calls[1], storage.result.to_dict("records")):
        assert generated_input.startswith(row["generated_prompt"])
        assert generated_input.endswith(row["text"])
    assert storage.result["generated_question"].tolist() == [
        f"Question {i}" for i in range(len(expected_source))
    ]
    assert result_keys == ["generated_question", "generated_answer"]
    pd.testing.assert_frame_equal(source, source_frame())


@pytest.mark.parametrize(
    "invalid_reply",
    [None, "null", '"a string"', '{"prompt": "value"}', '["valid", null]', "42"],
)
def test_malformed_prompt_response_does_not_shift_later_rows(invalid_reply):
    serving = StubServing([invalid_reply, '["Prompt B"]', '["Prompt C"]'])
    storage = MemoryStorage(source_frame())

    Text2QAGenerator(serving).run(storage)

    assert storage.result["id"].tolist() == ["B", "C"]
    assert storage.result["generated_prompt"].tolist() == ["Prompt B", "Prompt C"]


@pytest.mark.parametrize("reply", ["invalid JSON", "[]", None])
def test_no_valid_prompts_writes_empty_frame_without_qa_call(reply):
    source = source_frame()
    serving = StubServing([reply] * len(source))
    storage = MemoryStorage(source)

    result_keys = Text2QAGenerator(serving).run(
        storage,
        output_prompt_key="prompt",
        output_question_key="question",
        output_answer_key="answer",
    )

    assert len(serving.calls) == 1
    assert storage.result.empty
    assert storage.result.columns.tolist() == [
        "id", "text", "metadata", "prompt", "question", "answer"
    ]
    pd.testing.assert_frame_equal(
        storage.result[source.columns], source.iloc[:0].reset_index(drop=True)
    )
    assert result_keys == ["question", "answer"]


@pytest.mark.parametrize("response_count", [2, 4])
def test_rejects_prompt_response_count_mismatch(response_count):
    serving = StubServing(['["Prompt"]'] * response_count)
    storage = MemoryStorage(source_frame())

    with pytest.raises(RuntimeError, match=f"expected 3, got {response_count}"):
        Text2QAGenerator(serving).run(storage)

    assert storage.result is None
    assert len(serving.calls) == 1



def test_empty_source_writes_empty_frame():
    source = source_frame().iloc[:0]
    serving = StubServing([])
    storage = MemoryStorage(source)

    Text2QAGenerator(serving).run(storage)

    assert len(serving.calls) == 1
    assert storage.result.empty
    assert storage.result.columns.tolist() == [
        "id", "text", "metadata", "generated_prompt", "generated_question", "generated_answer"
    ]
    pd.testing.assert_frame_equal(
        storage.result[source.columns], source.reset_index(drop=True)
    )
