"""LLM-based hallucination sample evaluator.

Judges whether a finance answer is grounded in the given source text
(hallucination detection), producing a boolean flag and a reason.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融内容幻觉检测员。判断给定答案是否严格基于给定原文，"
    "重点检查：数字是否与原文一致、引用是否虚构、因果是否原文支持。\n"
    '只输出一个 JSON 对象：{"grounded": true/false, "reason": "..."}，'
    "不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance hallucination detector. Judge whether the answer is "
    "strictly grounded in the source text (numbers, citations, causal claims).\n"
    'Output a single JSON object: {"grounded": true/false, "reason": "..."} '
    "and nothing else."
)


@OPERATOR_REGISTRY.register()
class FinanceHallucinationSampleEvaluator(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "由 LLM 判定金融答案是否严格锚定原文（数字一致/引用真实/因果有据），"
                "输出幻觉标志与理由。"
            )
        return (
            "LLM-based hallucination detection for finance answers (numeric "
            "consistency, citation grounding, causal support) with reasons."
        )

    def _parse_verdict(self, text: str):
        import json

        for line in text.splitlines():
            line = line.strip().strip("`")
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    return bool(obj.get("grounded", False)), str(obj.get("reason", ""))
                except json.JSONDecodeError:
                    self.logger.warning("Skip malformed hallucination verdict line")
        return False, "parse-failed-default-ungrounded"

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        input_question: str = "question",
        input_answer: str = "answer",
        output_hallucination_flag: str = "hallucination",
        output_hallucination_reason: str = "hallucination_reason",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        flags, reasons = [], []
        for _, row in dataframe.iterrows():
            text = str(row.get(input_text, "") or "")
            question = str(row.get(input_question, "") or "")
            answer = str(row.get(input_answer, "") or "")
            if not answer.strip():
                flags.append(True)
                reasons.append("empty-answer")
                continue
            user_input = f"原文：{text}\n问题：{question}\n待检答案：{answer}"
            response = self.llm_serving.generate_from_input(
                user_inputs=[user_input],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            grounded, reason = self._parse_verdict(response)
            flags.append(not grounded)  # hallucination=True 表示未锚定
            reasons.append(reason)

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_hallucination_flag: flags,
                output_hallucination_reason: reasons,
            }
        )
        storage.write(result)
        return [output_hallucination_flag, output_hallucination_reason]
