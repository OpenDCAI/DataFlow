from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import pandas as pd

@OPERATOR_REGISTRY.register()
class KeywordTextGenerator(OperatorABC):
    """
    Generate narrative text based on provided keywords or topics.
    """
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "根据给定关键词生成文本。" if lang == "zh" else "Generate text based on given keywords."

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "instruction",  # changed default from "keyword" to "instruction"
        output_key: str = "generated_text",
        system_prompt: str = "You are a creative assistant. Write an engaging paragraph about the following topic: "
    ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        if self.input_key not in dataframe.columns:
            raise ValueError(
                f"Missing required column: {self.input_key}. Available columns: {list(dataframe.columns)}"
            )
        prompts = [system_prompt + str(x) for x in dataframe[self.input_key].astype(str).tolist()]
        generated_outputs = self.llm_serving.generate_from_input(prompts)
        dataframe[self.output_key] = generated_outputs
        storage.write(dataframe)
        return self.output_key


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
operator = KeywordTextGenerator(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
