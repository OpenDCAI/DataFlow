import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class ReasonStepAugment(OperatorABC):
    """
    ReasonStepAugment adds a detailed step-by-step reasoning chain to existing QA data.
    It does not create new rows; it only appends a new column with reasoning text.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template: str = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template or (
            "You are an expert in logical reasoning. "
            "Explain in detail how to arrive at the given answer step-by-step.\n\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Step-by-step reasoning:")

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "该算子在不新增数据行的情况下，为现有问答数据生成详细推理过程。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题字段名\n"
                "- input_answer_key: 答案字段名\n"
                "- output_key: 推理结果字段名\n")
        elif lang == "en":
            return (
                "This operator adds a detailed reasoning chain to existing QA records without creating new rows.\n\n"
                "Input Parameters:\n"
                "- input_question_key: Field name of the question\n"
                "- input_answer_key: Field name of the answer\n"
                "- output_key: Field name for generated reasoning\n")
        else:
            return "Adds step-by-step reasoning to QA data."

    def _validate_dataframe(self, df: pd.DataFrame, question_key: str, answer_key: str, output_key: str):
        missing = [k for k in [question_key, answer_key] if k not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if output_key in df.columns:
            raise ValueError(f"The output column '{output_key}' already exists and would be overwritten.")

    def _build_prompts(self, df: pd.DataFrame, question_key: str, answer_key: str) -> list:
        return [self.prompt_template.format(question=q, answer=a) for q, a in zip(df[question_key], df[answer_key])]

    def run(
        self,
        storage: DataFlowStorage,
        input_question_key: str = "question",
        input_answer_key: str = "golden_answers",  # Updated default to match dataset
        output_key: str = "reasoning_steps",
    ) -> list:
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_question_key, input_answer_key, output_key)
        prompts = self._build_prompts(df, input_question_key, input_answer_key)
        self.logger.info("Generating reasoning chains via LLM...")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="")
        df[output_key] = responses
        path = storage.write(df)
        self.logger.info(f"Reasoning augmentation complete. Saved to {path}")
        return [output_key]


# ======== Auto-generated runner ========
# from dataflow.utils.storage import FileStorage
# from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
# from dataflow.core import LLMServingABC

# if __name__ == "__main__":
#     # 1. FileStorage
#     storage = FileStorage(
#         first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl",
#         cache_path="./cache_local",
#         file_name_prefix="dataflow_cache_step",
#         cache_type="jsonl",
#     )

#     # 2. LLM-Serving
#     # -------- LLM Serving (Remote) --------
#     llm_serving = APILLMServing_request(
#         api_url="http://123.129.219.111:3000/v1/chat/completions",
#         key_name_of_api_key = 'DF_API_KEY',
#         model_name="gpt-4o",
#         max_workers=100,
#     )
#     # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# # 3. Instantiate operator
# operator = ReasonStepAugment(llm_serving=llm_serving, prompt_template="")

# # 4. Run
# operator.run(storage=storage.step())
