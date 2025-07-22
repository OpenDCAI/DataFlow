import json
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC


@OPERATOR_REGISTRY.register()
class MedicalDistractorGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子为医学多项选择题生成具有迷惑性的干扰选项，依据常见临床误区或医学伦理概念混淆构造。\n"
                "输入参数：\n"
                "- question_key: 题干字段名\n"
                "- answer_key: 正确答案字段名\n"
                "- output_distractor_key: 生成干扰项字段名\n"
            )
        else:
            return (
                "This operator generates realistic distractors for medical multiple-choice questions, using common clinical misconceptions or ethical confusions.\n\n"
                "Input Parameters:\n"
                "- question_key: field containing the question stem\n"
                "- answer_key: field containing the correct answer\n"
                "- output_distractor_key: field to store generated distractors\n"
            )

    def _validate_dataframe(self, df: pd.DataFrame):
        missing = [k for k in [self.question_key, self.answer_key] if k not in df.columns]
        conflict = [self.output_distractor_key] if self.output_distractor_key in df.columns else []
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The column {self.output_distractor_key} already exists and would be overwritten")

    def _build_prompt(self, df: pd.DataFrame):
        prompt_template = (
            "You are a medical board-style question writer. For the given question and its correct answer, "
            "generate exactly three plausible distractor options that reflect common clinical misconceptions "
            "or ethical confusions. Return only a JSON array of strings.\n\n"
            "Question: {q}\nCorrect answer: {a}\nDistractors:"
        )

        def extract_answer(ans):
            if isinstance(ans, list):
                return ans[0] if ans else ""
            return ans

        return [prompt_template.format(q=row[self.question_key], a=extract_answer(row[self.answer_key]))
                for _, row in df.iterrows()]

    def _parse(self, resp: str):
        try:
            start = resp.find("[")
            end = resp.rfind("]") + 1
            json_str = resp[start:end]
            distractors = json.loads(json_str)
            if isinstance(distractors, list):
                return distractors[:3]
        except Exception as e:
            self.logger.error(f"Failed to parse distractors: {e}")
        return ["", "", ""]

    def run(
        self,
        storage: DataFlowStorage,
        question_key: str = "question",
        answer_key: str = "golden_answers",
        output_distractor_key: str = "distractors",
    ):
        self.question_key, self.answer_key, self.output_distractor_key = (
            question_key,
            answer_key,
            output_distractor_key,
        )
        df = storage.read("dataframe")
        self._validate_dataframe(df)
        prompts = self._build_prompt(df)
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="")
        df[self.output_distractor_key] = [self._parse(r) for r in responses]
        output_file = storage.write(df)
        self.logger.info(f"Medical distractors saved to {output_file}")
        return [self.output_distractor_key]


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
# operator = MedicalDistractorGenerator(llm_serving=llm_serving)

# # 4. Run
# operator.run(storage=storage.step())
