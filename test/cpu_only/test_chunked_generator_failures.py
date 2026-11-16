"""Generation failures must not publish empty or partially validated output."""

from pathlib import Path

import pandas as pd
import pytest
import tiktoken

from dataflow.utils.storage import DataFlowStorage, FileStorage

pytestmark = pytest.mark.cpu


class CharacterEncoder:
    def encode(self, text):
        return list(text)


class FakeServing:
    def __init__(self, responses=None, error=None):
        self.responses = responses
        self.error = error
        self.calls = []

    def generate_from_input(self, user_inputs, **kwargs):
        self.calls.append((list(user_inputs), kwargs))
        if self.error is not None:
            raise self.error
        return self.responses


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://chunked-results"


@pytest.fixture
def generator_type(monkeypatch):
    # The public operator evaluates its default tokenizer during import. Avoid
    # a tokenizer download; an injected encoder is part of its existing API.
    with monkeypatch.context() as patch:
        patch.setattr(tiktoken, "get_encoding", lambda name: CharacterEncoder())
        from dataflow.operators.core_text import ChunkedPromptedGenerator
    return ChunkedPromptedGenerator


def failure_inputs(tmp_path):
    inputs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path, text in zip(inputs, ["abcd", "xy"]):
        path.write_text(text, encoding="utf-8")
    outputs = [path.with_name(f"{path.stem}_llm_output.txt") for path in inputs]
    for path in outputs:
        path.write_text("valuable previous output", encoding="utf-8")
    metadata = tmp_path / "source.jsonl"
    pd.DataFrame({"input_path": [str(path) for path in inputs]}).to_json(
        metadata, orient="records", lines=True
    )
    storage = FileStorage(str(metadata), cache_path=str(tmp_path / "cache")).step()
    return storage, inputs, outputs, metadata


def assert_unpublished(storage, inputs, outputs, metadata, original_metadata):
    assert [path.read_text(encoding="utf-8") for path in inputs] == ["abcd", "xy"]
    assert all(
        path.read_text(encoding="utf-8") == "valuable previous output"
        for path in outputs
    )
    assert metadata.read_bytes() == original_metadata
    assert not Path(storage._get_cache_file_path(1)).exists()


def test_serving_exception_propagates_without_overwriting_outputs(
    tmp_path, generator_type
):
    storage, inputs, outputs, metadata = failure_inputs(tmp_path)
    original_metadata = metadata.read_bytes()
    failure = RuntimeError("offline serving failure")
    serving = FakeServing(error=failure)
    generator = generator_type(serving, enc=CharacterEncoder(), max_chunk_len=2)
    with pytest.raises(RuntimeError) as raised:
        generator.run(storage, "input_path", "output_path")
    assert raised.value is failure
    assert_unpublished(storage, inputs, outputs, metadata, original_metadata)


@pytest.mark.parametrize(
    "responses,error_type",
    [
        (None, TypeError),
        ("abc", TypeError),
        ({0: "a", 1: "b", 2: "c"}, TypeError),
        (lambda: (value for value in ["a", "b", "c"]), TypeError),
        (["a", "b", None], TypeError),
        (["a", "b", 123], TypeError),
        (["a", "b"], RuntimeError),
        (["a", "b", "c", "extra"], RuntimeError),
    ],
    ids=[
        "none",
        "string",
        "mapping",
        "generator",
        "late-none",
        "late-number",
        "short",
        "long",
    ],
)
def test_invalid_response_batch_does_not_write_any_output(
    tmp_path, generator_type, responses, error_type
):
    storage, inputs, outputs, metadata = failure_inputs(tmp_path)
    original_metadata = metadata.read_bytes()
    if callable(responses):
        responses = responses()
    generator = generator_type(
        FakeServing(responses), enc=CharacterEncoder(), max_chunk_len=2
    )
    with pytest.raises(error_type):
        generator.run(storage, "input_path", "output_path")
    assert_unpublished(storage, inputs, outputs, metadata, original_metadata)


@pytest.mark.parametrize("response_type", [list, tuple])
def test_valid_batches_preserve_chunk_alignment_prompts_and_schema(
    tmp_path, generator_type, response_type
):
    inputs = [tmp_path / "first.txt", tmp_path / "empty.txt", tmp_path / "third.txt"]
    for path, text in zip(inputs, ["abcd", "", "xy"]):
        path.write_text(text, encoding="utf-8")
    metadata = tmp_path / "source.jsonl"
    pd.DataFrame({"input_path": [str(path) for path in inputs]}).to_json(
        metadata, orient="records", lines=True
    )
    storage = FileStorage(str(metadata), cache_path=str(tmp_path / "cache")).step()
    schema = {"type": "string"}
    serving = FakeServing(response_type(["first-a", "first-b", "", "third"]))
    generator = generator_type(
        serving,
        enc=CharacterEncoder(),
        max_chunk_len=2,
        system_prompt="instruction",
        json_schema=schema,
        separator="|",
    )
    assert generator.run(storage, "input_path", "output_path") == "output_path"
    assert serving.calls == [
        (
            ["instruction\nab", "instruction\ncd", "instruction\n", "instruction\nxy"],
            {"json_schema": schema},
        )
    ]
    result = pd.read_json(storage._get_cache_file_path(1), lines=True)
    assert [
        Path(path).read_text(encoding="utf-8") for path in result["output_path"]
    ] == ["first-a|first-b", "", "third"]
    assert [path.read_text(encoding="utf-8") for path in inputs] == ["abcd", "", "xy"]


@pytest.mark.parametrize("index", [[10, 30], [7, 7]])
def test_output_metadata_aligns_by_row_position_even_with_custom_indices(
    tmp_path, generator_type, index
):
    inputs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in inputs:
        path.write_text("x", encoding="utf-8")
    frame = pd.DataFrame({"input_path": [str(path) for path in inputs]}, index=index)
    storage = MemoryStorage(frame)
    generator = generator_type(FakeServing(["first", "second"]), enc=CharacterEncoder())
    generator.run(storage, "input_path", "output_path")
    expected = [str(path.with_name(f"{path.stem}_llm_output.txt")) for path in inputs]
    assert storage.result.index.tolist() == index
    assert storage.result["output_path"].tolist() == expected
    assert [Path(path).read_text(encoding="utf-8") for path in expected] == [
        "first",
        "second",
    ]
    pd.testing.assert_frame_equal(storage.dataframe, frame)


def test_zero_rows_skip_serving_and_publish_empty_output_metadata(generator_type):
    storage = MemoryStorage(pd.DataFrame({"input_path": pd.Series(dtype=str)}))
    serving = FakeServing(error=AssertionError("empty batch must not call serving"))
    generator = generator_type(serving, enc=CharacterEncoder())
    assert generator.run(storage, "input_path", "output_path") == "output_path"
    assert serving.calls == []
    assert storage.result.empty
    assert storage.result.columns.tolist() == ["input_path", "output_path"]
