"""Task-level examples: exact training output, invalid records and real pipelines."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from dataflow.operators.core_text import TrainingDataValidationFilter, TrainingMessagesConverter
from dataflow.pipeline import BatchedPipelineABC, PipelineABC, StreamBatchedPipelineABC
from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.storage import (
    BatchedFileStorage, FileStorage, LazyFileStorage, StreamBatchedFileStorage,
)
from dataflow.utils.training_data import TrainingDataError

pytestmark = pytest.mark.cpu


class MemoryStorage:
    def __init__(self, rows):
        self.frame = pd.DataFrame(deepcopy(rows))
        self.result = None

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.frame.copy()

    def write(self, frame):
        self.result = frame.copy()


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def result(call_id="c1", content="3"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


TOOL_MESSAGES = [
    user("先查两次，再回答"),
    {"role": "assistant", "content": None, "tool_calls": [call(), call("c2", "lookup", '{ "字": "词" }')]},
    result("c2", "词典"), result(),
    {"role": "assistant", "tool_calls": [call("c3", "finish", "{}")], "content": "核对结果"},
    result("c3", ""), assistant("结果是 3"),
    user("再说一遍"), assistant(" 3\n"),
]


ALPACA = {
    "id": "alpaca", "system": "你是助教", "history": [["1+1?", "2"], ["2+2?", "4"]],
    "instruction": "翻译", "input": " 你好\n", "output": " Hello\n", "metadata": {"split": "train"},
}
ALPACA_EXPECTED = [
    {"role": "system", "content": "你是助教"},
    {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
    {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
    {"role": "user", "content": "翻译\n\n 你好\n"}, {"role": "assistant", "content": " Hello\n"},
]
SHAREGPT = {
    "id": "sharegpt", "conversations": [
        {"from": "system", "value": "帮助用户"}, {"from": "human", "value": " Hi\n"},
        {"from": "gpt", "value": "你好"}, {"from": "human", "value": "再见"},
        {"from": "gpt", "value": " Bye "},
    ],
}
SHAREGPT_EXPECTED = [
    {"role": "system", "content": "帮助用户"}, {"role": "user", "content": " Hi\n"},
    {"role": "assistant", "content": "你好"}, {"role": "user", "content": "再见"},
    {"role": "assistant", "content": " Bye "},
]


@pytest.mark.parametrize("source_format,row,expected", [
    ("alpaca", ALPACA, ALPACA_EXPECTED),
    ("sharegpt", SHAREGPT, SHAREGPT_EXPECTED),
    ("messages", {"id": "tool", "messages": TOOL_MESSAGES}, TOOL_MESSAGES),
])
def test_conversion_produces_exact_training_messages_without_changing_source(source_format, row, expected):
    storage = MemoryStorage([row])
    before = deepcopy(storage.frame.to_dict("records"))
    TrainingMessagesConverter(source_format=source_format).run(storage, output_key="prepared")
    actual = storage.result.to_dict("records")[0]
    assert actual["prepared"] == expected
    assert {key: actual[key] for key in row} == row
    assert storage.frame.to_dict("records") == before
    if source_format == "messages":
        actual["prepared"][1]["tool_calls"][0]["function"]["arguments"] = "changed"
        assert actual["messages"] == TOOL_MESSAGES


def test_alpaca_defaults_allow_sparse_optional_columns_without_losing_history():
    rows = [ALPACA, {"instruction": "问", "output": "答"},
            {"instruction": "问2", "output": "答2", "input": "", "history": None, "system": None}]
    storage = MemoryStorage(rows)
    TrainingMessagesConverter(source_format="alpaca").run(storage)
    actual = storage.result["messages"].tolist()
    assert actual == [ALPACA_EXPECTED, [user("问"), assistant("答")], [user("问2"), assistant("答2")]]


def test_alpaca_supports_mapped_columns_and_does_not_merge_unrelated_fields():
    storage = MemoryStorage([{"q": "Q", "a": "A", "ctx": "C", "past": [["P", "R"]], "sys": "S", "reasoning": "keep separate"}])
    TrainingMessagesConverter(source_format="alpaca").run(
        storage, input_key="q", input_answer_key="a", input_context_key="ctx",
        input_history_key="past", input_system_key="sys", output_key="train",
    )
    assert storage.result.iloc[0]["train"] == [
        {"role": "system", "content": "S"}, user("P"), assistant("R"), user("Q\n\nC"), assistant("A"),
    ]
    assert storage.result.iloc[0]["reasoning"] == "keep separate"


@pytest.mark.parametrize("source_format,row", [
    ("qa", {"prompt": "Q", "answer": "A"}),
    ("alpaca", {"instruction": "Q", "output": "A"}),
    ("sharegpt", {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}),
    ("messages", {"messages": [user("Q"), assistant("A")]}),
])
def test_explicit_system_is_added_once(source_format, row):
    storage = MemoryStorage([row])
    TrainingMessagesConverter(system_prompt="S", source_format=source_format).run(storage)
    assert storage.result.iloc[0]["messages"] == [{"role": "system", "content": "S"}, user("Q"), assistant("A")]


def test_sharegpt_top_level_system_and_custom_source_key():
    storage = MemoryStorage([{"dialogue": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}], "system": "S"}])
    TrainingMessagesConverter(source_format="sharegpt").run(storage, input_key="dialogue")
    assert storage.result.iloc[0]["messages"] == [{"role": "system", "content": "S"}, user("Q"), assistant("A")]


@pytest.mark.parametrize("source_format,row,overrides,field,code", [
    ("alpaca", {"output": "A"}, {}, "instruction", "missing_field"),
    ("alpaca", {"instruction": "Q", "output": " "}, {}, "output", "empty_content"),
    ("alpaca", {"instruction": "Q", "output": "A", "input": []}, {}, "input", "invalid_type"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": {}}, {}, "history", "invalid_type"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": [["P"]]}, {}, "history[0]", "invalid_structure"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": [["P", " "]]}, {}, "history[0][1]", "empty_content"),
    ("alpaca", {"instruction": "Q", "output": "A", "system": 2}, {}, "system", "invalid_type"),
    ("alpaca", ALPACA, {"system_prompt": "override"}, "system", "system_conflict"),
    ("sharegpt", SHAREGPT, {"system_prompt": "override"}, "conversations[0]", "system_conflict"),
    ("sharegpt", {"system": "S", **SHAREGPT}, {}, "system", "system_conflict"),
    ("sharegpt", {}, {}, "conversations", "missing_field"),
    ("sharegpt", {"conversations": "Q"}, {}, "conversations", "invalid_type"),
    ("sharegpt", {"conversations": []}, {}, "conversations", "invalid_structure"),
    ("sharegpt", {"conversations": ["Q"]}, {}, "conversations[0]", "invalid_type"),
    ("sharegpt", {"conversations": [{"from": "function_call", "value": "{}"}]}, {}, "conversations[0].from", "invalid_role"),
    ("sharegpt", {"conversations": [{"from": [], "value": "Q"}]}, {}, "conversations[0].from", "invalid_role"),
    ("sharegpt", {"conversations": [{"from": "human", "value": " "}]}, {}, "conversations[0].value", "empty_content"),
    ("sharegpt", {"conversations": [{"from": "gpt", "value": "A"}]}, {}, "conversations[0].from", "invalid_structure"),
    ("sharegpt", {"conversations": [{"from": "human", "value": "Q", "tool_calls": []}]}, {}, "conversations[0]", "unsupported_fields"),
    ("messages", {"messages": [user("Q")]}, {}, "messages", "invalid_structure"),
])
def test_conversion_rejects_ambiguous_or_invalid_data_with_source_field(source_format, row, overrides, field, code):
    storage = MemoryStorage([row])
    with pytest.raises(TrainingDataError) as caught:
        TrainingMessagesConverter(source_format=source_format, **overrides).run(storage)
    assert (caught.value.field, caught.value.code) == (field, code)
    assert storage.result is None


@pytest.mark.parametrize("messages", [
    [user("Q"), assistant("A")], SHAREGPT_EXPECTED, ALPACA_EXPECTED, TOOL_MESSAGES,
    [user("Q"), {"role": "assistant", "content": "", "tool_calls": [call()]}, result(), assistant("A")],
])
def test_conversation_validator_accepts_complete_tasks_and_keeps_content(messages):
    storage = MemoryStorage([{"id": "record", "messages": messages}])
    operator = TrainingDataValidationFilter(message_profile="conversation")
    operator.run(storage)
    assert storage.result.to_dict("records") == [{"id": "record", "messages": messages, "training_data_valid": True}]
    assert operator.last_report == {"input": 1, "accepted": 1, "rejected": 0}


@pytest.mark.parametrize("messages,field,code", [
    (None, "messages", "invalid_type"),
    ([], "messages", "invalid_structure"),
    ([user("Q"), "A"], "messages[1]", "invalid_type"),
    ([{"role": [], "content": "Q"}], "messages[0].role", "invalid_role"),
    ([{"role": "human", "content": "Q"}], "messages[0].role", "invalid_role"),
    ([assistant("A")], "messages[0].role", "invalid_structure"),
    ([user("Q"), user("Q2")], "messages[1].role", "invalid_structure"),
    ([user("Q"), assistant("A"), assistant("A2")], "messages[2].role", "invalid_structure"),
    ([user("Q"), {"role": "system", "content": "S"}], "messages[1].role", "invalid_structure"),
    ([{"role": "system", "content": "S"}], "messages", "invalid_structure"),
    ([user("Q"), assistant("A"), user("Q2")], "messages", "invalid_structure"),
    ([user(" "), assistant("A")], "messages[0].content", "empty_content"),
    ([user("Q"), assistant("")], "messages[1].content", "empty_content"),
    ([user([{"type": "image_url"}]), assistant("A")], "messages[0].content", "invalid_type"),
    ([{**user("Q"), "tool_calls": [call()]}, assistant("A")], "messages[0]", "unsupported_fields"),
    ([user("Q"), {**assistant("A"), "reasoning_content": "R"}], "messages[1]", "unsupported_fields"),
    ([user("Q"), result()], "messages[1].tool_call_id", "unexpected_tool_response"),
    ([result(), user("Q"), assistant("A")], "messages[0].tool_call_id", "unexpected_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": []}], "messages[1].tool_calls", "invalid_structure"),
    ([user("Q"), {"role": "assistant", "tool_calls": {}}], "messages[1].tool_calls", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()], "content": 3}], "messages[1].content", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}], "messages[1].tool_calls", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, assistant("A")], "messages[2].role", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result("unknown"), assistant("A")], "messages[2].tool_call_id", "unexpected_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(), result(), assistant("A")], "messages[3].tool_call_id", "duplicate_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call(), call("c2")]}, result(), assistant("A")], "messages[3].role", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call(), call()]}, result(), assistant("A")], "messages[1].tool_calls[1].id", "duplicate_tool_call"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result()], "messages", "invalid_structure"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(content=None), assistant("A")], "messages[2].content", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, {"role": "tool", "content": "3"}, assistant("A")], "messages[2].tool_call_id", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(), {"role": "assistant", "tool_calls": [call()]}], "messages[3].tool_calls[0].id", "duplicate_tool_call"),
])
def test_conversation_validator_identifies_structural_and_pairing_errors(messages, field, code):
    storage = MemoryStorage([{"messages": messages}])
    with pytest.raises(TrainingDataError) as caught:
        TrainingDataValidationFilter(message_profile="conversation").run(storage)
    assert (caught.value.field, caught.value.code) == (field, code)
    assert storage.result is None


@pytest.mark.parametrize("broken_call,field,code", [
    ("call", "", "invalid_type"),
    ({**call(), "id": " "}, ".id", "empty_content"),
    ({**call(), "type": "other"}, ".type", "invalid_type"),
    ({**call(), "extra": 1}, "", "unsupported_fields"),
    ({**call(), "function": "add"}, ".function", "invalid_type"),
    ({**call(), "function": {"name": "add", "arguments": "{}", "extra": 1}}, ".function", "unsupported_fields"),
    (call(name=""), ".function.name", "empty_content"),
    (call(arguments={}), ".function.arguments", "invalid_type"),
    (call(arguments=""), ".function.arguments", "empty_content"),
    (call(arguments="{"), ".function.arguments", "invalid_json"),
    (call(arguments='{"x":NaN}'), ".function.arguments", "invalid_json"),
    (call(arguments='{"x":Infinity}'), ".function.arguments", "invalid_json"),
    (call(arguments="[]"), ".function.arguments", "invalid_type"),
    (call(arguments="null"), ".function.arguments", "invalid_type"),
])
def test_tool_call_payload_validation(broken_call, field, code):
    messages = [user("Q"), {"role": "assistant", "tool_calls": [broken_call]}, result(), assistant("A")]
    with pytest.raises(TrainingDataError) as caught:
        TrainingDataValidationFilter(message_profile="conversation").run(MemoryStorage([{"messages": messages}]))
    assert (caught.value.field, caught.value.code) == ("messages[1].tool_calls[0]" + field, code)


def test_default_validator_still_uses_single_turn_profile():
    with pytest.raises(TrainingDataError):
        TrainingDataValidationFilter().run(MemoryStorage([{"messages": SHAREGPT_EXPECTED}]))


@pytest.mark.parametrize("operator,kwargs", [
    (TrainingMessagesConverter, {"source_format": "auto"}),
    (TrainingDataValidationFilter, {"message_profile": "auto"}),
    (TrainingDataValidationFilter, {"schema": "dpo", "message_profile": "conversation"}),
])
def test_invalid_profile_is_a_configuration_error(operator, kwargs):
    with pytest.raises(ValueError):
        operator(**kwargs)


@pytest.mark.parametrize("source_format,kwargs", [
    ("qa", {"input_history_key": "history"}),
    ("sharegpt", {"input_answer_key": "answer"}),
    ("messages", {"input_context_key": "context"}),
    ("alpaca", {"input_history_key": ""}),
])
def test_unused_or_blank_mappings_are_not_silently_ignored(tmp_path, source_format, kwargs):
    storage = MemoryStorage([ALPACA])
    with pytest.raises(ValueError):
        TrainingMessagesConverter(source_format=source_format, on_invalid="quarantine", errors_path=tmp_path / "errors").run(storage, **kwargs)
    assert storage.result is None
    assert not (tmp_path / "errors").exists()


def test_strict_conversation_validation_does_not_write_a_valid_prefix():
    storage = MemoryStorage([{"messages": TOOL_MESSAGES}, {"messages": [user("Q")]}])
    with pytest.raises(TrainingDataError, match="row 2"):
        TrainingDataValidationFilter(message_profile="conversation").run(storage)
    assert storage.result is None


@pytest.mark.parametrize("operator,field", [
    (TrainingMessagesConverter(), "prompt"),
    (TrainingMessagesConverter(source_format="alpaca"), "instruction"),
    (TrainingDataValidationFilter(message_profile="conversation"), "messages"),
])
def test_empty_object_is_rejected_instead_of_disappearing(operator, field):
    storage = MemoryStorage([{}])
    with pytest.raises(TrainingDataError) as caught:
        operator.run(storage)
    assert (caught.value.field, caught.value.code) == (field, "missing_field")
    assert storage.result is None


class ConversationPipeline(PipelineABC):
    def __init__(self, storage, source_format, errors):
        super().__init__()
        self.storage, self.source_format = storage, source_format
        self.converter = TrainingMessagesConverter(source_format=source_format, on_invalid="quarantine", errors_path=errors)
        self.validator = TrainingDataValidationFilter(message_profile="conversation")

    def forward(self):
        if self.source_format == "alpaca":
            self.converter.run(self.storage.step(), input_key="instruction", input_answer_key="output")
        else:
            self.converter.run(self.storage.step(), input_key="conversations" if self.source_format == "sharegpt" else "messages")
        self.validator.run(self.storage.step(), input_key="messages")


class BatchConversationPipeline(ConversationPipeline, BatchedPipelineABC):
    _compiled_forward = BatchedPipelineABC._compiled_forward


class StreamConversationPipeline(ConversationPipeline, StreamBatchedPipelineABC):
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward


@pytest.mark.parametrize("storage_cls,pipeline_cls", [
    (FileStorage, ConversationPipeline), (LazyFileStorage, ConversationPipeline),
    (BatchedFileStorage, BatchConversationPipeline), (StreamBatchedFileStorage, StreamConversationPipeline),
])
@pytest.mark.parametrize("source_format,valid,invalid,expected", [
    ("alpaca", ALPACA, {**ALPACA, "output": ""}, ALPACA_EXPECTED),
    ("sharegpt", SHAREGPT, {**SHAREGPT, "conversations": [{"from": "human", "value": "unfinished"}]}, SHAREGPT_EXPECTED),
    ("messages", {"messages": TOOL_MESSAGES}, {"messages": TOOL_MESSAGES[:2]}, TOOL_MESSAGES),
])
def test_task_output_and_rejections_in_file_lazy_batch_and_stream_pipeline(
    tmp_path, storage_cls, pipeline_cls, source_format, valid, invalid, expected,
):
    # First batch is wholly invalid; later batches must still preserve all turns.
    rows = [{**invalid, "id": f"bad{i}"} for i in range(2)] + [{**valid, "id": f"ok{i}"} for i in range(3)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    before = source.read_bytes()
    extra = {"save_on_exit": False} if storage_cls is LazyFileStorage else {}
    storage = storage_cls(str(source), cache_path=str(tmp_path / "cache"), **extra)
    errors = tmp_path / "errors.jsonl"
    pipeline = pipeline_cls(storage, source_format, errors)
    pipeline.compile()
    if storage_cls in (BatchedFileStorage, StreamBatchedFileStorage):
        pipeline.forward(batch_size=2, resume_from_last=False)
    else:
        pipeline.forward()
    if storage_cls is LazyFileStorage:
        storage.flush_all()
    actual = pd.read_json(storage._get_cache_file_path(2), lines=True).to_dict("records")
    assert [row["id"] for row in actual] == ["ok0", "ok1", "ok2"]
    assert [row["messages"] for row in actual] == [expected, expected, expected]
    assert all(row["training_data_valid"] is True for row in actual)
    assert source.read_bytes() == before
    assert [json.loads(line)["row"] for line in errors.read_text().splitlines()] == [1, 2]


def test_tool_quarantine_reports_physical_source_line_and_exact_field(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("bad JSON\n" + json.dumps({"messages": TOOL_MESSAGES}) + "\n" + json.dumps({
        "messages": [user("secret input"), {"role": "assistant", "tool_calls": [call()]}, result("wrong"), assistant("A")],
    }) + "\n", encoding="utf-8")
    storage = FileStorage(str(source), cache_path=str(tmp_path / "cache"), jsonl_read_policy=JSONLReadPolicy(tmp_path / "read-errors"))
    errors = tmp_path / "validation-errors"
    operator = TrainingDataValidationFilter(message_profile="conversation", on_invalid="quarantine", errors_path=errors)
    operator.run(storage.step())
    report = json.loads(errors.read_text())
    assert report["field"] == "messages[2].tool_call_id"
    assert report["code"] == "unexpected_tool_response"
    assert report["source_location"]["line"] == 3
    assert "secret input" not in errors.read_text()
    assert operator.last_report == {"input": 2, "accepted": 1, "rejected": 1}
    assert pd.read_json(storage._get_cache_file_path(1), lines=True).iloc[0]["messages"] == TOOL_MESSAGES


class ValidationPipeline(PipelineABC):
    def __init__(self, storage, errors):
        super().__init__()
        self.storage = storage
        self.validator = TrainingDataValidationFilter(
            message_profile="conversation", on_invalid="quarantine", errors_path=errors,
        )

    def forward(self):
        self.validator.run(self.storage.step(), input_key="messages")


class StreamValidationPipeline(ValidationPipeline, StreamBatchedPipelineABC):
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward


@pytest.mark.parametrize("storage_cls,pipeline_cls", [
    (FileStorage, ValidationPipeline), (StreamBatchedFileStorage, StreamValidationPipeline),
])
def test_validator_pipeline_excludes_incomplete_calls_and_keeps_later_conversations(tmp_path, storage_cls, pipeline_cls):
    rows = [
        {"id": "missing-return", "messages": TOOL_MESSAGES[:2]},
        {"id": "unfinished", "messages": [user("unfinished")]},
        {"id": "good", "messages": TOOL_MESSAGES},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    storage = storage_cls(str(source), cache_path=str(tmp_path / "cache"))
    errors = tmp_path / "errors.jsonl"
    pipeline = pipeline_cls(storage, errors)
    pipeline.compile()
    if storage_cls is StreamBatchedFileStorage:
        pipeline.forward(batch_size=2, resume_from_last=False)
    else:
        pipeline.forward()
    actual = pd.read_json(storage._get_cache_file_path(1), lines=True).to_dict("records")
    assert actual == [{"id": "good", "messages": TOOL_MESSAGES, "training_data_valid": True}]
    rejected = [json.loads(line) for line in errors.read_text().splitlines()]
    assert [(row["row"], row["field"], row["code"]) for row in rejected] == [
        (1, "messages[1].tool_calls", "missing_tool_response"),
        (2, "messages", "invalid_structure"),
    ]
