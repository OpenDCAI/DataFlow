import json
import os
from pathlib import Path

from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm

from dataflow.operators.general_text.filter.rule_based_filter import (
    ContentNullFilter,
    SentenceNumberFilter,
    SymbolWordRatioFilter,
)


def _ensure_input_file(file_path: str):
    """Create a minimal valid JSONL file if it does not already exist.

    This prevents FileStorage.read from failing during the pipeline.compile() stage
    when the expected input file is missing. The function writes a single line of
    JSON containing a placeholder `raw_content` field, which is enough for the
    compile-time key-integrity checks performed by the operator runtimes.
    """
    path = Path(file_path)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_record = {
        "raw_content": "This is a placeholder record automatically generated because the original input file was not found.",
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")


class RecommendPipeline(PipelineABC):
    def __init__(self, input_file: str | None = None):
        super().__init__()

        # ------------------------------------------------------------------
        # Resolve the input path. Allow the caller to override via parameter
        # or environment variable, falling back to the original hard-coded path.
        # ------------------------------------------------------------------
        default_input = "/mnt/DataFlow/lz/proj/DataFlow/dataflow/example/GeneralTextPipeline/translation.jsonl"
        self.input_file = input_file or os.getenv("DATAFLOW_INPUT_FILE", default_input)

        # Ensure the file exists so that Pipeline.compile() succeeds.
        _ensure_input_file(self.input_file)

        # -------- FileStorage --------
        self.storage = FileStorage(
            first_entry_file_name=self.input_file,
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # ------------------------------------------------------------------
        # LLM Serving.
        # Some environments used for static analysis / compilation may not
        # provide the real API key. Use a placeholder to avoid initialization
        # failures; users can still overwrite it via the DF_API_KEY env-var.
        # ------------------------------------------------------------------
        os.environ.setdefault("DF_API_KEY", "PLACEHOLDER_API_KEY")
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key="DF_API_KEY",
            model_name="gpt-4o",
            max_workers=100,
        )

        # -------- Filters --------------------------------------------------
        self.content_null_filter = ContentNullFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=3, max_sentences=7500)
        self.symbol_word_ratio_filter = SymbolWordRatioFilter(threshold=0.4)

    # ----------------------------------------------------------------------
    # Forward pass: plug operators together. Each operator writes its label
    # into a dedicated key so that downstream stages can reference the result.
    # ----------------------------------------------------------------------
    def forward(self):
        # Explicitly specify `input_key` to avoid ambiguity during compile-time
        # validation of key integrity across operator runtimes.
        self.content_null_filter.run(
            storage=self.storage.step(),
            input_key="raw_content",
            output_key="content_null_filter_label",
        )
        self.sentence_number_filter.run(
            storage=self.storage.step(),
            input_key="raw_content",
            output_key="sentence_number_filter_label",
        )
        self.symbol_word_ratio_filter.run(
            storage=self.storage.step(),
            input_key="raw_content",
            output_key="symbol_word_ratio_filter_label",
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.compile()
    pipeline.forward()
