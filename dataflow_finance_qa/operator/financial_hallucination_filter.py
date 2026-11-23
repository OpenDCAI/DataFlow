"""Financial hallucination filter operator.

Filters generated QA pairs by asking an LLM judge whether the answer is
fully grounded in the source text (no invented facts/numbers).
Keeps only grounded rows.
"""

import json

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

JUDGE_PROMPT_ZH = (
    "你是金融 QA 数据质检员。判断答案是否完全由来源文本支撑（无编造事实/数字/结论）。\n"
    "仅输出 JSON：{\"grounded\": true/false, \"reason\": \"...\"}"
)


@OPERATOR_REGISTRY.register()
class FinancialHallucinationFilter(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "过滤金融 QA 对：由 LLM 逐条判断答案是否锚定来源文本，"
                "剔除含编造事实或虚构数字的样本（幻觉过滤）。"
            )
        return (
            "Filter financial QA pairs with an LLM judge on answer grounding; "
            "drop samples with hallucinated facts or numbers."
        )

    def _judge(self, text: str, question: str, answer: str) -> bool:
        user_input = f"来源文本：\n{text}\n\n问题：{question}\n答案：{answer}"
        response = self.llm_serving.generate_from_input(
            user_inputs=[user_input], system_prompt=JUDGE_PROMPT_ZH
        )[0]
        try:
            return bool(json.loads(response.strip().strip("`")).get("grounded"))
        except (json.JSONDecodeError, AttributeError):
            self.logger.warning("Judge returned malformed JSON, drop sample")
            return False

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        input_question: str = "question",
        input_answer: str = "answer",
    ):
        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")
        mask = []
        for text, question, answer in zip(
            dataframe[input_text],
            dataframe[input_question],
            dataframe[input_answer],
        ):
            if not question or not answer:
                mask.append(False)
                continue
            mask.append(self._judge(text, question, answer))
        filtered = dataframe[mask]
        self.logger.info(
            f"Grounded {sum(mask)}/{len(mask)} QA pairs passed hallucination filter"
        )
        storage.write(filtered)
        return [input_question, input_answer]
