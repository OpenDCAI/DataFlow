from __future__ import annotations

import pandas as pd
import pytest

from dataflow.operators.core_text import PromptedGenerator


pytestmark = pytest.mark.cpu


class MemoryStorage:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.writes = []

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.writes.append(dataframe.copy())
        return "memory://prompted-generator"


class FailingServing:
    def __init__(self):
        self.calls = []

    def generate_from_input(self, user_inputs=None, system_prompt=None, json_schema=None):
        self.calls.append(
            {
                "user_inputs": list(user_inputs),
                "system_prompt": system_prompt,
                "json_schema": json_schema,
            }
        )
        raise RuntimeError("controlled serving failure")


class SuccessfulServing:
    def generate_from_input(self, user_inputs=None, system_prompt=None, json_schema=None):
        return [f"generated:{value}" for value in user_inputs]


def source_frame():
    return pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "raw_content": ["first", "", "third"],
        },
        index=[3, 7, 11],
    )


def test_serving_failure_propagates_and_does_not_write_success_output():
    serving = FailingServing()
    storage = MemoryStorage(source_frame())
    operator = PromptedGenerator(
        llm_serving=serving,
        system_prompt="system",
        user_prompt="Generate: ",
        json_schema={"type": "string"},
    )

    with pytest.raises(RuntimeError, match="controlled serving failure"):
        operator.run(
            storage=storage,
            input_key="raw_content",
            output_key="generated",
        )

    assert storage.writes == []
    assert serving.calls == [
        {
            "user_inputs": ["Generate: first", "Generate: third"],
            "system_prompt": "system",
            "json_schema": {"type": "string"},
        }
    ]


def test_successful_run_still_writes_aligned_output():
    storage = MemoryStorage(source_frame())
    operator = PromptedGenerator(
        llm_serving=SuccessfulServing(),
        user_prompt="Generate: ",
    )

    output_key = operator.run(
        storage=storage,
        input_key="raw_content",
        output_key="generated",
    )

    assert output_key == "generated"
    assert len(storage.writes) == 1
    assert list(storage.writes[0]["id"]) == ["A", "B", "C"]
    assert list(storage.writes[0]["generated"]) == [
        "generated:Generate: first",
        "",
        "generated:Generate: third",
    ]
