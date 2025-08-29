from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class SummaryGenerator(OperatorABC):
    """
    SummaryGenerator automatically produces concise summaries from long input texts
    using the provided LLM serving backend.
    """
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        # use tokenizer for length control when available
        self.tokenizer = getattr(llm_serving, "tokenizer", None)
        self.max_tokens = 4096  # hard limit to avoid context overflow

    @staticmethod
    def get_desc(lang: str = "zh"):
        return (
            "自动对长文本进行摘要生成。" if lang == "zh" else "Automatically generate summaries from long texts."
        )

    def _build_prompt(self, text: str, lang: str = "en") -> str:
        if lang == "zh":
            return f"请对以下内容进行简明扼要的摘要：\n\n{text}\n\n摘要："
        return f"Please provide a concise summary of the following text.\n\n{text}\n\nSummary:"

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "instruction",
        output_key: str = "summary",
        lang: str = "en",
    ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"SummaryGenerator loaded dataframe with {len(dataframe)} rows")

        llm_inputs = []
        for _, row in dataframe.iterrows():
            raw_content = row.get(self.input_key, "")
            if not raw_content:
                llm_inputs.append("")
                continue

            if self.tokenizer:
                tokens = self.tokenizer.encode(raw_content, truncation=True, max_length=self.max_tokens)
                raw_content = self.tokenizer.decode(tokens, skip_special_tokens=True)
            prompt = self._build_prompt(raw_content, lang=lang)
            llm_inputs.append(prompt)

        try:
            self.logger.info("Calling LLM to generate summaries ...")
            summaries = self.llm_serving.generate_from_input(llm_inputs)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise e

        dataframe[self.output_key] = summaries
        storage.write(dataframe)
        return [output_key]


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
        model_name="deepseek-v3",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = SummaryGenerator(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
