"""Explicit local-JSONL execution; not a registered field-transform operator."""
from pathlib import Path
from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.training_data import check_keys
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError, RecordResult


class RecordedGenerationRunner:
    """Process independent records against an explicit immutable local source.

    Source/output paths belong to this task runner, not to an Operator.run
    signature. export_results is explicitly a successful-record view; the ledger
    retains failures. This runner is not a generic Pipeline wrapper.
    """
    def __init__(self, llm_serving, ledger_dir, run_config, system_prompt="You are a helpful assistant", parse_response=None):
        if not getattr(llm_serving, "record_results", False):
            raise ValueError("Recorded generation requires Serving(record_results=True)")
        if not isinstance(run_config, dict):
            raise ValueError("run_config must declare model/parser configuration")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text")
        self.last_report = {}
        self.llm_serving = llm_serving
        self.ledger_dir = ledger_dir
        self.run_config = run_config
        self.system_prompt = system_prompt
        self.parse_response = parse_response

    def run(self, source, successful_output, input_key="raw_content", output_key="generated_content",
            only_failed=False, read_policy=None, outcomes_output=None, status_key="generation_outcome"):
        check_keys(input_key, output_key, status_key)
        if input_key == output_key or status_key in {input_key, output_key}:
            raise ValueError("Input, output and status fields must be distinct")
        source = Path(source).resolve()
        successful_output = Path(successful_output).resolve()
        outcomes_output = (Path(outcomes_output).resolve() if outcomes_output is not None else
                           successful_output.with_name(successful_output.stem + ".outcomes.jsonl"))
        state = Path(self.ledger_dir).resolve()
        if successful_output == outcomes_output or any(
                path == source or path.is_relative_to(state) for path in (successful_output, outcomes_output)):
            raise ValueError("Exports must differ from source, each other and ledger state")
        # Configuration/schema errors are discovered before any model side effect.
        for row, location, code in JSONLReadPolicy._records(source):
            if code is None and (output_key in row or status_key in row):
                raise ValueError("Output and status fields must not overwrite source columns")
        config = {"declared": self.run_config, "system_prompt": self.system_prompt,
                  "model": getattr(self.llm_serving, "model_name", None),
                  "api_url": getattr(self.llm_serving, "api_url", None),
                  "model_options": getattr(self.llm_serving, "configs", {}),
                  "input_key": input_key, "output_key": output_key, "status_key": status_key,
                  "request_budget_seconds": getattr(self.llm_serving, "request_budget_seconds", None),
                  "max_retries": getattr(self.llm_serving, "max_retries", None)}
        def process(row):
            text = row.get(input_key)
            if not isinstance(text, str) or not text.strip():
                raise RecordProcessingError("invalid_input", "input", attempts=0)
            if output_key in row:
                raise RecordProcessingError("output_field_conflict", "input", attempts=0)
            responses = self.llm_serving.generate_from_input([text], self.system_prompt)
            event = self.llm_serving.last_results[0]
            if responses[0] is None or (isinstance(responses[0], str) and not responses[0].strip()):
                raise RecordProcessingError(event.get("error_code") or "empty_response", "request", event["attempts"])
            try:
                output = self.parse_response(responses[0]) if self.parse_response else responses[0]
            except ValueError:
                raise RecordProcessingError("invalid_model_output", "parse", event["attempts"])
            return RecordResult({output_key: output}, event["attempts"])
        with RecordLedger(self.ledger_dir, source, config, read_policy) as ledger:
            ledger.process(process, only_failed=only_failed)
            ledger.export_results(successful_output)
            ledger.export_outcomes(outcomes_output, output_key, status_key)
            self.last_report = ledger.report()
            return self.last_report
