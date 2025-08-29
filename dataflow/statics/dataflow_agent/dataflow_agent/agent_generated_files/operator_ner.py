from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import pandas as pd
import json
import re


@OPERATOR_REGISTRY.register()
class NamedEntityRecognizerLLM(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用大语言模型从输入文本中识别人名、地名、组织名三类命名实体，并输出对应列。"
            )
        else:
            return (
                "Uses an LLM to recognize three types of named entities (persons, locations, organizations) in the input text and outputs them as separate columns."
            )

    def _validate_dataframe(self, df: pd.DataFrame):
        if self.input_key not in df.columns:
            raise ValueError(f"Column {self.input_key} not found in dataframe")
        for col in ["persons", "locations", "organizations"]:
            if col in df.columns:
                raise ValueError(f"Column {col} already exists and would be overwritten")

    def _build_prompts(self, df: pd.DataFrame):
        template = (
            "You are an information extraction assistant. Extract all PERSON, LOCATION and ORGANIZATION entities from the following text. "
            "Return a JSON object with exactly three keys: 'persons', 'locations', 'organizations', each containing a list of unique strings. "
            "If none found, use empty list. Text:\n\"\"\"\n{content}\n\"\"\""
        )
        return [template.format(content=str(content)) for content in df[self.input_key]]

    def run(self, storage: DataFlowStorage, input_key: str = "instruction"):
        self.input_key = input_key
        df = storage.read("dataframe")
        self._validate_dataframe(df)

        # Initialize result columns to ensure correct dtype and avoid broadcasting issues
        for col in ["persons", "locations", "organizations"]:
            df[col] = pd.Series([[] for _ in range(len(df))], dtype=object)

        prompts = self._build_prompts(df)
        responses = self.llm_serving.generate_from_input(prompts)

        for idx, resp in zip(df.index, responses):
            try:
                cleaned = re.sub(r"^```json\s*|\s*```$", "", resp.strip(), flags=re.DOTALL)
                parsed = json.loads(cleaned)
                df.at[idx, "persons"] = parsed.get("persons", [])
                df.at[idx, "locations"] = parsed.get("locations", [])
                df.at[idx, "organizations"] = parsed.get("organizations", [])
            except Exception as e:
                self.logger.error(f"Failed to parse NER result for row {idx}: {e}")
                df.at[idx, "persons"] = []
                df.at[idx, "locations"] = []
                df.at[idx, "organizations"] = []

        output_file = storage.write(df)
        self.logger.info(f"Named entity recognition results saved to {output_file}")
        return ["persons", "locations", "organizations"]


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
operator = NamedEntityRecognizerLLM(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
