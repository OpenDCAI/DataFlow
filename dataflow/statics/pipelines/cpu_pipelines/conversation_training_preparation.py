"""Offline Alpaca, ShareGPT and tool-conversation preparation with expected outputs."""

import argparse
import json
from pathlib import Path

from dataflow.operators.text_sft import AlpacaMessagesGenerator, AlpacaStructureSampleEvaluator
from dataflow.operators.conversations import (ShareGPTMessagesGenerator, ShareGPTStructureSampleEvaluator,
    ConversationStructureSampleEvaluator, ConversationStructureFilter)
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage


class ConversationPreparationPipeline(PipelineABC):
    def __init__(self, source, output_dir, source_format):
        super().__init__()
        self.storage = FileStorage(str(source), cache_path=str(output_dir / "cache"))
        self.source_format = source_format
        if source_format == "alpaca":
            self.source_check = AlpacaStructureSampleEvaluator()
            self.converter = AlpacaMessagesGenerator()
        elif source_format == "sharegpt":
            self.source_check = ShareGPTStructureSampleEvaluator()
            self.converter = ShareGPTMessagesGenerator()
        elif source_format == "messages":
            self.source_check = ConversationStructureSampleEvaluator()
        else:
            raise ValueError("source_format must be alpaca, sharegpt or messages")
        self.keep_valid = ConversationStructureFilter()
        self.validator = ConversationStructureSampleEvaluator()

    def forward(self):
        if self.source_format == "alpaca":
            self.source_check.run(self.storage.step(), input_context_key="input", input_history_key="history", input_system_key="system")
        else:
            self.source_check.run(self.storage.step())
        self.keep_valid.run(self.storage.step(), input_key="conversation_valid" if self.source_format == "messages" else "source_valid")
        if self.source_format == "alpaca":
            self.converter.run(self.storage.step(), input_context_key="input", input_history_key="history", input_system_key="system")
        elif self.source_format == "sharegpt":
            self.converter.run(self.storage.step())
        self.validator.run(self.storage.step(), output_valid_key="final_valid", output_error_key="final_error")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Fixed tasks and expected answers, without models or platform credentials.
    alpaca = {
        "id": "alpaca", "instruction": "翻译", "input": "你好", "output": "Hello",
        "history": [["1+1?", "2"]], "system": "帮助用户",
    }
    alpaca_expected = [
        {"role": "system", "content": "帮助用户"},
        {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
        {"role": "user", "content": "翻译\n\n你好"}, {"role": "assistant", "content": "Hello"},
    ]
    sharegpt = {"id": "sharegpt", "conversations": [
        {"from": "human", "value": "Hi"}, {"from": "gpt", "value": "你好"},
        {"from": "human", "value": "Bye"}, {"from": "gpt", "value": "再见"},
    ]}
    sharegpt_expected = [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "再见"},
    ]
    tool_messages = [
        {"role": "user", "content": "计算 1+2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "add-1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
        ]},
        {"role": "tool", "tool_call_id": "add-1", "content": "3"},
        {"role": "assistant", "content": "结果为 3"},
    ]
    cases = [
        ("alpaca", alpaca, {**alpaca, "id": "bad", "output": ""}, alpaca_expected, "output", "empty_content"),
        ("sharegpt", sharegpt, {"id": "bad", "conversations": sharegpt["conversations"][:-1]}, sharegpt_expected, "conversations", "invalid_structure"),
        ("messages", {"id": "tool", "messages": tool_messages}, {"id": "bad", "messages": tool_messages[:2]}, tool_messages, "messages[1].tool_calls", "missing_tool_response"),
    ]
    summary = {}
    for source_format, good, bad, expected, error_field, error_code in cases:
        root = output / source_format
        root.mkdir()
        source = root / "source.jsonl"
        _write_jsonl(source, [good, bad])
        original = source.read_bytes()
        pipeline = ConversationPreparationPipeline(source, root, source_format)
        pipeline.compile()
        pipeline.forward()
        prepared = [json.loads(line) for line in Path(pipeline.storage._get_cache_file_path(3 if source_format == "messages" else 4)).read_text(encoding="utf-8").splitlines()]
        assert len(prepared) == 1 and prepared[0]["id"] == good["id"]
        assert prepared[0]["messages"] == expected
        assert prepared[0]["final_valid"] is True
        evaluated = [json.loads(line) for line in Path(pipeline.storage._get_cache_file_path(1)).read_text(encoding="utf-8").splitlines()]
        error_key = "conversation_error" if source_format == "messages" else "source_error"
        errors = [row[error_key] for row in evaluated if row[error_key] is not None]
        assert len(evaluated) == 2 and len(errors) == 1
        assert (errors[0]["row"], errors[0]["field"], errors[0]["code"]) == (2, error_field, error_code)
        assert source.read_bytes() == original
        _write_jsonl(root / "training.jsonl", [{"messages": row["messages"]} for row in prepared])
        summary[source_format] = {"input": 2, "accepted": 1, "rejected": 1, "matches_expected": True}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified Alpaca, ShareGPT and tool conversations: 3 accepted, 3 quarantined. Output: {output}")


if __name__ == "__main__":
    main()
