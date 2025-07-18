from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import pandas as pd

@OPERATOR_REGISTRY.register()
class MedicalQuestionContextualizer(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None,
                 difficulty: str = "medium",
                 output_key: str = "enhanced_question"):
        """
        Enrich medical exam questions with additional clinical context while
        keeping the original intent. Optionally adjust difficulty.
        difficulty: "easy" | "medium" | "hard" (only used to guide prompt)
        output_key: name of the column to store enhanced questions.
        """
        self.llm_serving = llm_serving
        self.difficulty = difficulty.lower()
        if self.difficulty not in ["easy", "medium", "hard"]:
            self.difficulty = "medium"
        self.output_key = output_key
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于在保持原题意的前提下，为医学考试题目加入额外临床背景信息，并可根据需要调整题目难度。\n"
                "输入参数：\n"
                "- llm_serving: 大语言模型服务实例\n"
                "- difficulty: 目标难度级别（easy/medium/hard），默认medium\n"
                "- output_key: 输出列名，默认enhanced_question\n"
                "运行时参数：input_key 指定待处理题目所在列名\n"
                "输出：新增一列包含增强后的题目文本。"
            )
        else:
            return (
                "Adds supplemental clinical context to medical exam questions while preserving intent, and tunes question difficulty.\n"
                "Input Params:\n"
                "- llm_serving: LLM serving instance\n"
                "- difficulty: target difficulty level (easy/medium/hard), default medium\n"
                "- output_key: output column name, default enhanced_question\n"
                "Runtime param input_key identifies the column containing questions.\n"
                "Output: new column with enriched questions."
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        if input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column {self.output_key} already exists and would be overwritten.")

    def _build_prompt(self, question: str) -> str:
        base_template = (
            "You are a medical educator. Rewrite the following medical exam question by adding realistic clinical context (patient age, symptoms, brief history) while preserving the original intent and answer. "
            "Make the overall difficulty {diff} for examinees. Return ONLY the rewritten question.\n\n"
            "Original question: \n{q}"
        )
        return base_template.format(diff=self.difficulty, q=question)

    def run(self, storage: DataFlowStorage, input_key: str = "question"):
        """Process the dataframe stored in DataFlowStorage and add contextualized questions.\n
        Args:
            storage (DataFlowStorage): Storage object used in the data flow runtime.
            input_key (str, optional): Column name that contains the original questions. Defaults to "question".
        """
        dataframe: pd.DataFrame = storage.read("dataframe")
        self._validate_dataframe(dataframe, input_key)

        prompts = [self._build_prompt(q) for q in dataframe[input_key]]
        responses = self.llm_serving.generate_from_input(prompts)

        dataframe[self.output_key] = responses
        output_file = storage.write(dataframe)
        self.logger.info(f"Enhanced questions saved to {output_file}")
        return [self.output_key]


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
# operator = MedicalQuestionContextualizer(llm_serving=llm_serving, difficulty='medium', output_key='enhanced_question')

# # 4. Run
# operator.run(storage=storage.step())
