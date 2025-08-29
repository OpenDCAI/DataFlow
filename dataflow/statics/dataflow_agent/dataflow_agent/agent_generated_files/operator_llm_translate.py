from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import pandas as pd

@OPERATOR_REGISTRY.register()
class En2ZhTranslator(OperatorABC):
    # common column names that may store the English text
    DEFAULT_INPUT_KEY_CANDIDATES = ["instruction", "english_text", "output", "source"]

    def __init__(self, llm_serving: LLMServingABC = None,
                 prompt_prefix: str = "Translate the following English text into Chinese:"):
        self.llm_serving = llm_serving
        self.prompt_prefix = prompt_prefix
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于将英文文本翻译为中文。\n\n"
                "输入参数：\n"
                "- input_key：英文文本字段名（可自动推断，默认为 instruction 或其他常见列名）\n\n"
                "输出参数：\n"
                "- chinese_text：翻译后的中文文本"
            )
        else:
            return (
                "This operator translates English text into Chinese.\n\n"
                "Input Parameters:\n"
                "- input_key: Column name containing English text (automatically inferred if omitted)\n\n"
                "Output Parameters:\n"
                "- chinese_text: Translated Chinese text"
            )

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    def _infer_input_key(self, df: pd.DataFrame, input_key: str | None):
        """Return the provided key or try to guess from common candidates."""
        if input_key:  # user specified
            return input_key
        for key in self.DEFAULT_INPUT_KEY_CANDIDATES:
            if key in df.columns:
                return key
        raise ValueError(
            f"Unable to locate English text column. Available columns: {list(df.columns)}. "
            "Please specify input_key explicitly.")

    def _validate_dataframe(self, df: pd.DataFrame):
        if self.input_key not in df.columns:
            raise ValueError(
                f"Missing required column: {self.input_key}. "
                f"Available columns: {list(df.columns)}")
        if self.output_key in df.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    def _build_prompts(self, df: pd.DataFrame):
        return [f"{self.prompt_prefix}\n\n{text}" for text in df[self.input_key].fillna("")]

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------
    def run(self, storage: DataFlowStorage,
            input_key: str | None = None,
            output_key: str = "chinese_text"):
        # 1. load data
        df = storage.read("dataframe")

        # 2. decide column names
        self.input_key = self._infer_input_key(df, input_key)
        self.output_key = output_key

        # 3. basic validation
        self._validate_dataframe(df)

        # 4. build prompts & call LLM
        prompts = self._build_prompts(df)
        responses = self.llm_serving.generate_from_input(prompts)
        df[self.output_key] = [resp.strip() if resp else "" for resp in responses]

        # 5. persist & return
        output_file = storage.write(df)
        self.logger.info(f"Translation results saved to {output_file}")
        return [self.output_key]


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
    # -------- LLM Serving (Local) --------
    llm_serving = LocalModelLLMServing_vllm(
        hf_model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        vllm_tensor_parallel_size=1,
        vllm_max_tokens=8192,
        hf_local_dir="local",
    )

# 3. Instantiate operator
operator = En2ZhTranslator(llm_serving=llm_serving, prompt_prefix='Translate the following English text into Chinese:')

# 4. Run
operator.run(storage=storage.step())
