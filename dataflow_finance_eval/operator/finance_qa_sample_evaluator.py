"""Deterministic finance QA sample evaluator.

Scores model answers against gold answers: normalized exact/fuzzy match for
factual questions, numeric extraction with relative tolerance for numeric
questions. Pure logic, no LLM required.
"""

import re

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@OPERATOR_REGISTRY.register()
class FinanceQASampleEvaluator(OperatorABC):
    def __init__(self, numeric_tolerance: float = 0.01, fuzzy_threshold: float = 0.8):
        super().__init__()
        self.numeric_tolerance = numeric_tolerance  # 数值相对容差
        self.fuzzy_threshold = fuzzy_threshold  # 字符重叠率阈值
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "确定性评测金融问答：事实题用归一化精确/模糊匹配，"
                "数值题按相对容差比较（自动从答案中提取数字与单位），无需 LLM。"
            )
        return (
            "Deterministic finance QA evaluation: normalized exact/fuzzy matching "
            "for factual answers, relative-tolerance numeric comparison; no LLM."
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"\s+", "", str(text or ""))
        text = text.replace("，", "").replace("。", "").replace("；", "")
        return text.lower()

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        """以较短一方的字符集为分母，衡量金标要点被答案覆盖的程度。"""
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        return len(common) / min(len(set(a)), len(set(b)))

    def _score_one(self, answer: str, gold: str) -> float:
        ans, ref = self._normalize(answer), self._normalize(gold)
        if not ans or not ref:
            return 0.0
        if ans == ref:
            return 1.0
        # 数值比较：提取全部数字，数量一致且每个都在容差内 → 1.0
        ans_nums, ref_nums = _NUM_RE.findall(ans), _NUM_RE.findall(ref)
        if ans_nums and ref_nums and len(ans_nums) == len(ref_nums):
            try:
                if all(
                    abs(float(x) - float(y)) <= self.numeric_tolerance * max(abs(float(y)), 1e-12)
                    for x, y in zip(ans_nums, ref_nums)
                ):
                    return 1.0
            except ValueError:
                pass
        # 含数字的答案禁用字符模糊匹配：数字置换/失真必须判错（如 1.32%→2.31%）
        if not ans_nums and not ref_nums and self._char_overlap(ans, ref) >= self.fuzzy_threshold:
            return 1.0
        return 0.0

    def run(
        self,
        storage: DataFlowStorage,
        input_answer: str = "answer",
        input_gold_answer: str = "gold_answer",
        output_qa_score: str = "qa_score",
        output_qa_correct: str = "qa_correct",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        scores = []
        for _, row in dataframe.iterrows():
            scores.append(
                self._score_one(
                    str(row.get(input_answer, "") or ""),
                    str(row.get(input_gold_answer, "") or ""),
                )
            )

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_qa_score: scores,
                output_qa_correct: [s >= 1.0 for s in scores],
            }
        )
        storage.write(result)
        return [output_qa_score, output_qa_correct]
