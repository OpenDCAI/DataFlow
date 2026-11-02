"""Offline training preparation example; run with --output-dir /tmp/prepared."""

import argparse
import json
from pathlib import Path

from dataflow.operators.text_sft import QAStructureSampleEvaluator, QAMessagesGenerator
from dataflow.operators.conversations import ValidationFlagFilter, ConversationStructureSampleEvaluator
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage


class TrainingPreparationPipeline(PipelineABC):
    def __init__(self, source, output_dir):
        super().__init__()
        self.storage = FileStorage(str(source), cache_path=str(output_dir / "cache"))
        self.source_check = QAStructureSampleEvaluator()
        self.keep_valid = ValidationFlagFilter()
        self.converter = QAMessagesGenerator()
        self.validator = ConversationStructureSampleEvaluator()

    def forward(self):
        self.source_check.run(self.storage.step(), input_key="question", input_answer_key="answer")
        self.keep_valid.run(self.storage.step(), input_key="source_valid")
        self.converter.run(self.storage.step(), input_key="question", input_answer_key="answer")
        self.validator.run(self.storage.step())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Refuse reuse: error ledgers append and must not be mistaken for a fresh run.
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source.jsonl"
    rows = [
        {"id": "one", "question": "1+1?", "answer": "2"},
        {"id": "two", "question": "你好", "answer": "你好！"},
        {"id": "bad", "question": "missing answer", "answer": ""},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    pipeline = TrainingPreparationPipeline(source, output)
    pipeline.compile()
    pipeline.forward()
    # Source fields and validation flag are retained in the cache. Export only
    # the requested training format here, without changing the Storage contract.
    cached = Path(pipeline.storage._get_cache_file_path(4))
    prepared = [json.loads(line) for line in cached.read_text(encoding="utf-8").splitlines()]
    training = output / "training.jsonl"
    training.write_text("".join(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n" for row in prepared), encoding="utf-8")
    assert len(prepared) == 2
    evaluated = [json.loads(line) for line in Path(pipeline.storage._get_cache_file_path(1)).read_text().splitlines()]
    assert len(evaluated) == 3 and sum(not row['source_valid'] for row in evaluated) == 1
    print(f"Prepared 2 records; quarantined 1 record. Output: {training}")


if __name__ == "__main__":
    main()
