import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

@OPERATOR_REGISTRY.register()
class TextQuestionGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, num_questions: int = 3):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.num_questions = num_questions
        if self.num_questions not in range(1, 6):
            raise ValueError("num_questions must be an integer between 1 and 5 (inclusive)")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于根据输入文本生成相关问题。\n\n"
                "输入参数：\n"
                "- input_key: 包含待分析文本的字段名\n"
                "- output_key: 输出生成问题的字段名\n"
                "- num_questions: 每段文本生成的问题数量 (1-5)\n\n"
                "输出参数：\n"
                "- output_key: 生成的问题字符串"
            )
        else:
            return (
                "This operator generates relevant questions for given text passages.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the text\n"
                "- output_key: Field name for generated questions\n"
                "- num_questions: Number of questions per passage (1-5)\n\n"
                "Output Parameters:\n"
                "- output_key: Generated questions as a single string"
            )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if output_key in df.columns:
            raise ValueError(f"Column {output_key} already exists and would be overwritten")

    def _build_prompts(self, df: pd.DataFrame, input_key: str):
        prompts = []
        for text in df[input_key]:
            prompt = (
                f"Please read the following text and generate {self.num_questions} relevant questions.\n\n"
                f"Text:\n{text}\n\nQuestions:"
            )
            prompts.append(prompt)
        return prompts

    def _resolve_input_key(self, df: pd.DataFrame, input_key: str) -> str:
        """Automatically resolve a valid input_key if the provided one is absent."""
        if input_key in df.columns:
            return input_key
        # Try common alternatives found in the dataset
        for alt_key in ["instruction", "output", "source", "golden_answer"]:
            if alt_key in df.columns:
                self.logger.warning(
                    f"Input column '{input_key}' not found. Falling back to '{alt_key}'."
                )
                return alt_key
        # If nothing matches, keep original (validation will raise)
        return input_key

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "text",
        output_key: str = "generated_questions"
    ):
        df = storage.read("dataframe")
        # Try to resolve a valid input key if the default is missing
        input_key = self._resolve_input_key(df, input_key)
        self._validate_dataframe(df, input_key, output_key)
        prompts = self._build_prompts(df, input_key)
        responses = self.llm_serving.generate_from_input(prompts)
        df[output_key] = [r.strip() for r in responses]
        output_file = storage.write(df)
        self.logger.info(f"Generated questions saved to {output_file}")
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
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = TextQuestionGenerator(llm_serving=llm_serving, num_questions=3)

# 4. Run
operator.run(storage=storage.step())
