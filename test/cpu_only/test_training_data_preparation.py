"""Public operators must preserve training content and work in real pipelines."""

import json
from pathlib import Path

import pandas as pd
import pytest

from dataflow.operators.core_text import (
    TrainingDataValidationFilter,
    TrainingMessagesConverter,
    TrainingPreferenceConverter,
)
from dataflow.pipeline import PipelineABC, StreamBatchedPipelineABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, LazyFileStorage, StreamBatchedFileStorage
from dataflow.utils.training_data import TrainingDataError

pytestmark = pytest.mark.cpu


class MemoryStorage:
    def __init__(self, rows):
        self.frame = pd.DataFrame(rows)
        self.result = None

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.frame.copy()

    def write(self, frame):
        self.result = frame.copy()


def write_source(tmp_path, rows):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return source


def test_messages_preserve_whitespace_unicode_and_metadata():
    storage = MemoryStorage([{"question": " 你好\n", "response": "答案\n ", "id": "a"}])
    operator = TrainingMessagesConverter(system_prompt="Be helpful")
    assert operator.run(storage, input_key="question", input_answer_key="response") == ["messages"]
    row = storage.result.to_dict("records")[0]
    assert row == {
        "id": "a", "question": " 你好\n", "response": "答案\n ",
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": " 你好\n"},
            {"role": "assistant", "content": "答案\n "},
        ],
    }
    assert operator.last_report == {"input": 1, "accepted": 1, "rejected": 0}


def test_preference_preserves_both_answers_and_shared_prompt():
    storage = MemoryStorage([{"q": " Q\n", "good": "A", "bad": " B ", "id": 4}])
    outputs = TrainingPreferenceConverter().run(
        storage, input_key="q", input_chosen_key="good", input_rejected_key="bad",
    )
    assert outputs == ["prompt", "chosen", "rejected"]
    row = storage.result.to_dict("records")[0]
    assert (row["prompt"], row["chosen"], row["rejected"], row["id"]) == (" Q\n", "A", " B ", 4)
    checked = MemoryStorage([row])
    TrainingDataValidationFilter(schema="dpo").run(checked)
    assert checked.result["training_data_valid"].tolist() == [True]


@pytest.mark.parametrize("record,field,code", [
    ({"prompt": "q"}, "answer", "missing_field"),
    ({"prompt": "q", "answer": None}, "answer", "invalid_type"),
    ({"prompt": "q", "answer": 42}, "answer", "invalid_type"),
    ({"prompt": "q", "answer": " \n"}, "answer", "empty_content"),
    ({"prompt": ["q"], "answer": "a"}, "prompt", "invalid_type"),
])
def test_conversion_errors_are_locatable_and_do_not_write(record, field, code):
    storage = MemoryStorage([record])
    with pytest.raises(TrainingDataError, match="row 1") as caught:
        TrainingMessagesConverter().run(storage)
    assert (caught.value.field, caught.value.code) == (field, code)
    assert storage.result is None


def test_late_strict_error_does_not_commit_valid_prefix():
    storage = MemoryStorage([{"prompt": "q", "answer": "a"}, {"prompt": "q", "answer": ""}])
    with pytest.raises(TrainingDataError, match="row 2"):
        TrainingMessagesConverter().run(storage)
    assert storage.result is None


def test_quarantine_reports_positions_across_batches_without_raw_content(tmp_path):
    errors = tmp_path / "errors.jsonl"
    operator = TrainingMessagesConverter(on_invalid="quarantine", errors_path=errors)
    storage = MemoryStorage([{"prompt": "secret", "answer": ""}, {"prompt": "q", "answer": "a"}])
    storage.batch_step, storage.batch_size = 2, 2
    operator.run(storage)
    assert len(storage.result) == 1
    report = json.loads(errors.read_text())
    assert (report["row"], report["field"], report["code"]) == (5, "answer", "empty_content")
    assert "secret" not in errors.read_text()
    assert operator.last_report == {"input": 2, "accepted": 1, "rejected": 1}
    storage.batch_step = 3
    operator.run(storage)
    assert [json.loads(line)["row"] for line in errors.read_text().splitlines()] == [5, 7]
    # The last report is per invocation, not a silent accumulation across jobs.
    assert operator.last_report["input"] == 2


@pytest.mark.parametrize("operator", [TrainingMessagesConverter, TrainingPreferenceConverter, TrainingDataValidationFilter])
def test_quarantine_requires_error_destination(operator):
    with pytest.raises(ValueError, match="errors_path"):
        operator(on_invalid="quarantine")


@pytest.mark.parametrize("value", [None, "", "  ", 3, ["text"]])
def test_dpo_rejected_is_required_text(value):
    storage = MemoryStorage([{"prompt": "q", "chosen": "a", "rejected": value}])
    with pytest.raises(TrainingDataError) as caught:
        TrainingDataValidationFilter(schema="dpo").run(storage)
    assert caught.value.field == "rejected"
    assert storage.result is None


@pytest.mark.parametrize("messages,field,code", [
    (None, "messages", "invalid_type"),
    ([], "messages", "invalid_structure"),
    ([{"role": "user", "content": "q"}, {"role": "tool", "content": "a"}], "messages[1].role", "invalid_role"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": " "}], "messages[1].content", "empty_content"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": []}], "messages[1]", "unsupported_fields"),
    ([{"role": "user", "content": "q"}, "answer"], "messages[1]", "invalid_type"),
])
def test_messages_profile_reports_nested_error(messages, field, code):
    with pytest.raises(TrainingDataError) as caught:
        TrainingDataValidationFilter().run(MemoryStorage([{"messages": messages}]))
    assert (caught.value.field, caught.value.code) == (field, code)


def test_all_quarantined_retains_output_schema(tmp_path):
    storage = MemoryStorage([{"prompt": "q", "answer": ""}])
    TrainingMessagesConverter(on_invalid="quarantine", errors_path=tmp_path / "errors.jsonl").run(storage)
    assert storage.result.empty
    assert storage.result.columns.tolist() == ["prompt", "answer", "messages"]
    # The next operator can process an empty batch without losing schema.
    checked = MemoryStorage([])
    checked.frame = storage.result
    TrainingDataValidationFilter().run(checked)
    assert checked.result.empty
    assert checked.result.columns.tolist() == ["prompt", "answer", "messages", "training_data_valid"]


def test_error_ledger_cannot_append_to_source(tmp_path):
    source = write_source(tmp_path, [{"prompt": "q", "answer": ""}])
    original = source.read_bytes()
    storage = FileStorage(str(source), cache_path=str(tmp_path / "cache")).step()
    with pytest.raises(ValueError, match="source file"):
        TrainingMessagesConverter(on_invalid="quarantine", errors_path=source).run(storage)
    assert source.read_bytes() == original


class PreparationPipeline(PipelineABC):
    def __init__(self, storage, schema):
        super().__init__()
        self.storage = storage
        self.schema = schema
        self.converter = TrainingMessagesConverter() if schema == "messages" else TrainingPreferenceConverter()
        self.validator = TrainingDataValidationFilter(schema=schema)

    def forward(self):
        if self.schema == "messages":
            self.converter.run(self.storage.step(), input_key="q", input_answer_key="good")
            self.validator.run(self.storage.step(), input_key="messages")
        else:
            self.converter.run(self.storage.step(), input_key="q", input_chosen_key="good", input_rejected_key="bad")
            self.validator.run(self.storage.step(), input_prompt_key="prompt", input_chosen_key="chosen", input_rejected_key="rejected")


class StreamPreparationPipeline(PreparationPipeline, StreamBatchedPipelineABC):
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward


@pytest.mark.parametrize("schema", ["messages", "dpo"])
@pytest.mark.parametrize("storage_cls", [FileStorage, LazyFileStorage, StreamBatchedFileStorage])
def test_real_pipeline_matches_direct_calls_and_preserves_source(tmp_path, schema, storage_cls):
    rows = [{"q": f"问题{i}", "good": f"答案{i}", "bad": f"错误{i}", "id": i} for i in range(5)]
    source = write_source(tmp_path, rows)
    original = source.read_bytes()
    kwargs = {"save_on_exit": False} if storage_cls is LazyFileStorage else {}
    storage = storage_cls(str(source), cache_path=str(tmp_path / "cache"), **kwargs)
    cls = StreamPreparationPipeline if storage_cls is StreamBatchedFileStorage else PreparationPipeline
    pipeline = cls(storage, schema)
    pipeline.compile()
    if storage_cls is StreamBatchedFileStorage:
        pipeline.forward(batch_size=2, resume_from_last=False)
    else:
        pipeline.forward()
    if storage_cls is LazyFileStorage:
        storage.flush_all()
    actual = pd.read_json(storage._get_cache_file_path(2), lines=True).to_dict("records")

    first = MemoryStorage(rows)
    if schema == "messages":
        TrainingMessagesConverter().run(first, input_key="q", input_answer_key="good")
    else:
        TrainingPreferenceConverter().run(first, input_key="q", input_chosen_key="good", input_rejected_key="bad")
    second = MemoryStorage(first.result.to_dict("records"))
    TrainingDataValidationFilter(schema=schema).run(second)
    assert actual == second.result.to_dict("records")
    assert source.read_bytes() == original


def test_public_imports_are_registered():
    for cls in (TrainingMessagesConverter, TrainingPreferenceConverter, TrainingDataValidationFilter):
        assert OPERATOR_REGISTRY.get(cls.__name__) is cls


def test_invalid_configuration_does_not_quarantine_or_write(tmp_path):
    storage = MemoryStorage([{"prompt": "q", "chosen": "a", "rejected": "b"}])
    with pytest.raises(ValueError, match="distinct"):
        TrainingPreferenceConverter().run(storage, output_rejected_key="chosen")
    with pytest.raises(ValueError, match="overwrite"):
        TrainingDataValidationFilter(schema="dpo").run(storage, output_key="chosen")
    with pytest.raises(ValueError, match="nonblank"):
        TrainingMessagesConverter().run(storage, output_key="")
    assert storage.result is None


@pytest.mark.parametrize("schema", ["messages", "dpo"])
def test_stream_quarantine_matches_unbatched_output_and_error_locations(tmp_path, schema):
    rows = [{"q": f"q{i}", "good": "a" if i % 2 else "", "bad": "b", "id": i} for i in range(5)]
    source = write_source(tmp_path, rows)
    outputs, ledgers = [], []
    for label, storage_cls, pipeline_cls in [
        ("file", FileStorage, PreparationPipeline),
        ("stream", StreamBatchedFileStorage, StreamPreparationPipeline),
    ]:
        root = tmp_path / label
        storage = storage_cls(str(source), cache_path=str(root / "cache"))
        pipeline = pipeline_cls(storage, schema)
        pipeline.converter.on_invalid = "quarantine"
        pipeline.converter.errors_path = root / "errors.jsonl"
        pipeline.compile()
        if label == "stream":
            pipeline.forward(batch_size=2, resume_from_last=False)
        else:
            pipeline.forward()
        outputs.append(pd.read_json(storage._get_cache_file_path(2), lines=True).to_dict("records"))
        ledgers.append([json.loads(line)["row"] for line in (root / "errors.jsonl").read_text().splitlines()])
    assert outputs[0] == outputs[1]
    assert [row["id"] for row in outputs[0]] == [1, 3]
    assert ledgers == [[1, 3, 5], [1, 3, 5]]
