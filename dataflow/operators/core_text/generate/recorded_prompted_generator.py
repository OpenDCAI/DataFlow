"""Opt-in per-record failures and replayable results for text generation."""

from pathlib import Path

import pandas as pd

from dataflow.core import OperatorABC
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError, RecordResult
from dataflow.utils.pipeline_checkpoint import state_lock
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class RecordedPromptedGenerator(OperatorABC):
    def __init__(self, llm_serving, ledger_dir, run_config, system_prompt="You are a helpful assistant", parse_response=None):
        super().__init__()
        if not getattr(llm_serving, "record_results", False):
            raise ValueError("Recorded generation requires Serving(record_results=True)")
        self.llm_serving = llm_serving
        self.ledger_dir = ledger_dir
        self.run_config = run_config
        self.system_prompt = system_prompt
        self.parse_response = parse_response
        self.last_report = {}
        self._source_inspection = None

    def configuration(self, input_key="raw_content", output_key="generated_content"):
        return {"declared": self.run_config, "system_prompt": self.system_prompt,
                "model": getattr(self.llm_serving, "model_name", None),
                "api_url": getattr(self.llm_serving, "api_url", None),
                "model_options": getattr(self.llm_serving, "configs", {}),
                "input_key": input_key, "output_key": output_key,
                "request_budget_seconds": getattr(self.llm_serving, "request_budget_seconds", None),
                "max_retries": getattr(self.llm_serving, "max_retries", None)}

    def processor(self, input_key="raw_content", output_key="generated_content"):
        """Create the same processor for ledger.process(..., only_failed=True)."""
        def process(row):
            text = row.get(input_key)
            if not isinstance(text, str) or not text.strip():
                raise RecordProcessingError("invalid_input", "input", attempts=0)
            responses = self.llm_serving.generate_from_input([text], self.system_prompt)
            event = self.llm_serving.last_results[0]
            if responses[0] is None or (isinstance(responses[0], str) and not responses[0].strip()):
                raise RecordProcessingError(event.get("error_code") or "empty_response", "request", event["attempts"])
            try:
                result = self.parse_response(responses[0]) if self.parse_response is not None else responses[0]
            except ValueError:
                raise RecordProcessingError("invalid_model_output", "parse", event["attempts"])
            return RecordResult({output_key: result}, event["attempts"])
        return process

    def replay_failed(self, source, output_path, input_key="raw_content", output_key="generated_content", read_policy=None):
        """Retry recorded failures and rebuild output without appending duplicates."""
        with RecordLedger(self.ledger_dir, source, self.configuration(input_key, output_key), read_policy) as ledger:
            ledger.process(self.processor(input_key, output_key), only_failed=True)
            ledger.export_results(output_path)
            self.last_report = ledger.report()
        return self.last_report

    def run(self, storage, input_key="raw_content", output_key="generated_content"):
        source = Path(storage._get_cache_file_path(storage.operator_step)).resolve()
        if not source.is_file() or source.suffix != ".jsonl":
            raise ValueError("Recorded generation requires a materialized local JSONL input")
        if Path(storage._get_cache_file_path(storage.operator_step + 1)).resolve() == source:
            raise ValueError("Recorded generation output must differ from its input")
        frame = storage.read("dataframe")
        locations = frame.attrs.get("dataflow_source_locations")
        offset = (getattr(storage, "batch_size", None) or 0) * getattr(storage, "batch_step", 0)
        policy = getattr(storage, "jsonl_read_policy", None) if storage.operator_step == 0 else None
        configuration = self.configuration(input_key, output_key)
        processor = self.processor(input_key, output_key)
        rows = []
        with state_lock(self.ledger_dir), RecordLedger(self.ledger_dir, source, configuration, policy, self._source_inspection) as ledger:
            self._source_inspection = ledger.source_inspection
            # pandas returns [] for a DataFrame containing rows but no columns.
            # Empty objects still need terminal invalid_input outcomes.
            records = frame.to_dict("records") if len(frame.columns) else [{} for _ in range(len(frame))]
            for index, row in enumerate(records):
                line = locations[index]["line"] if locations is not None else offset + index + 1
                outcome = ledger.get(line)
                if outcome is None:
                    try:
                        result = processor(row)
                    except RecordProcessingError as exc:
                        ledger.record(line, "failed", stage=exc.stage, code=exc.code, attempts=exc.attempts)
                    else:
                        ledger.record(line, "success", result=result.updates, attempts=result.attempts)
                    outcome = ledger.get(line)
                if outcome["status"] == "success":
                    rows.append({**row, **outcome["result"]})
            columns = list(dict.fromkeys([*frame.columns, output_key]))
            storage.write(pd.DataFrame(rows, columns=columns))
            ledger.publish()
            self.last_report = ledger.report()
        return [output_key]

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return "逐条文本生成，保存源记录关联、失败原因和结果，支持独立任务的失败补跑。"
        return "Generate per-record text with source-linked outcomes and replayable failures."
