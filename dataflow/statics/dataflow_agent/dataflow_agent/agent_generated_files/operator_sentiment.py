import pandas as pd
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class SentimentAnalyzer(OperatorABC):
    """
    Binary sentiment analyzer (positive/negative) using an LLM backend.
    """

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子基于大语言模型对文本进行情感分析，输出正面或负面标签。\n\n"
                "输入参数：\n"
                "- input_key: 包含文本内容的字段名\n"
                "- output_sentiment_key: 输出情感标签字段名（positive/negative）\n"
            )
        elif lang == "en":
            return (
                "This operator performs binary sentiment analysis (positive/negative) using an LLM.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the text\n"
                "- output_sentiment_key: Field name for the sentiment label (positive/negative)\n"
            )
        else:
            return "Binary sentiment analysis operator."

    def _validate_dataframe(self, df: pd.DataFrame):
        missing = [] if self.input_key in df.columns else [self.input_key]
        conflict = [self.output_sentiment_key] if self.output_sentiment_key in df.columns else []
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The column {self.output_sentiment_key} already exists and would be overwritten."
            )

    def _build_prompts(self, df: pd.DataFrame):
        return [
            f"Classify the sentiment of the following text as Positive or Negative.\nText: {row[self.input_key]}\nSentiment:"
            for _, row in df.iterrows()
        ]

    @staticmethod
    def _parse_label(response: str):
        m = re.search(r"positive|negative", response, re.IGNORECASE)
        return "positive" if m and m.group(0).lower() == "positive" else "negative"

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str | None = None,
        output_sentiment_key: str = "sentiment",
    ):
        """Run the operator.

        The function will try to automatically find a suitable input column if
        `input_key` is None or not found in the dataframe.
        """
        df = storage.read("dataframe")

        # Determine the input column
        candidate_keys = []
        if input_key:
            candidate_keys.append(input_key)
        candidate_keys.extend(["text", "instruction", "output", "golden_answer", "source"])
        self.input_key = next((k for k in candidate_keys if k in df.columns), None)
        if self.input_key is None:
            raise ValueError(
                f"None of the candidate input keys {candidate_keys} found in dataframe."
            )

        self.output_sentiment_key = output_sentiment_key

        # Validation
        self._validate_dataframe(df)

        # Build prompts and get responses
        prompts = self._build_prompts(df)
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="")
        labels = [self._parse_label(r) for r in responses]

        # Write results
        df[self.output_sentiment_key] = labels
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_sentiment_key]


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="../example_data/DataflowAgent/agent_test_data.json",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Remote) --------
    llm_serving = APILLMServing_request(
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = SentimentAnalyzer(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
