from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import numpy as np
import pandas as pd
import re
from typing import List


@OPERATOR_REGISTRY.register()
class ReasoningAnswerNgramFilter(OperatorABC):
    """
    适配中英文的 n-gram 重复度过滤算子：
    - 若文本包含中文字符：按“字符 n-gram”计算重复度；
    - 否则：按“英文单词 n-gram”计算重复度。
    重复度得分 = unique_ngrams_count / total_ngrams ∈ [0, 1]，
    数值越小表示重复越严重。
    """

    def __init__(
        self,
        min_score: float = 0.1,
        max_score: float = 1.0,
        ngrams: int = 5,
    ):
        self.logger = get_logger()

        # 基本参数赋值
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.ngrams = int(ngrams)

        # 参数合法性校正，不抛异常，只打日志并兜底修正
        if not (0.0 <= self.min_score <= 1.0):
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] min_score={self.min_score} 不在 [0, 1]，重置为 0.0"
            )
            self.min_score = 0.0

        if not (0.0 <= self.max_score <= 1.0):
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] max_score={self.max_score} 不在 [0, 1]，重置为 1.0"
            )
            self.max_score = 1.0

        if self.min_score > self.max_score:
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] min_score({self.min_score}) > max_score({self.max_score})，交换二者数值"
            )
            self.min_score, self.max_score = self.max_score, self.min_score

        if self.ngrams <= 0:
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] ngrams={self.ngrams} 非正数，重置为 5"
            )
            self.ngrams = 5

    # ---------------- 公共说明 ----------------
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子基于 n-gram 重复率过滤答案，支持中英文：\n"
                "- 若文本包含中文：使用“字符 n-gram”来度量重复度；\n"
                "- 若文本不含中文：使用“英文单词 n-gram”来度量重复度。\n\n"
                "输入参数：\n"
                "- min_score：最小可接受分数（0~1，越低表示越重复）\n"
                "- max_score：最大可接受分数（0~1）\n"
                "- ngrams：n-gram 的窗口大小\n\n"
                "输出行为：\n"
                "- 计算每条样本的重复度分数 repetition_score ∈ [0, 1]\n"
                "- 仅保留分数在 [min_score, max_score] 区间内的样本。"
            )
        elif lang == "en":
            return (
                "This operator filters answers based on n-gram repetition scores and supports both Chinese and English.\n"
                "- If the text contains Chinese characters: use character-level n-grams.\n"
                "- Otherwise: use word-level n-grams for English.\n\n"
                "Input Parameters:\n"
                "- min_score: Minimum acceptable score (0~1, lower means more repetition)\n"
                "- max_score: Maximum acceptable score (0~1)\n"
                "- ngrams: Size of the n-gram window\n\n"
                "Behavior:\n"
                "- Compute repetition_score ∈ [0, 1] for each sample\n"
                "- Keep only samples whose scores fall in [min_score, max_score]."
            )
        else:
            return "ReasoningAnswerNgramFilter detects repetition via n-gram scores."

    # ---------------- 内部工具函数 ----------------
    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """判断文本中是否包含中文字符。"""
        if not isinstance(text, str):
            return False
        return re.search(r"[\u4e00-\u9fff]", text) is not None

    def _build_char_ngrams(self, text: str) -> List[str]:
        """
        针对包含中文的文本，使用“字符 n-gram”。

        处理步骤：
        1. 转为小写；
        2. 仅保留：中文字符 + 英文字母 + 数字；
        3. 删除空白等非内容字符；
        4. 按连续 n 个“字符”构造 n-gram。
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        text = text.lower()
        # 保留中文、英文字母、数字
        text = re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", text)

        chars = list(text)
        if len(chars) < self.ngrams:
            return []

        ngrams = [
            "".join(chars[i : i + self.ngrams])
            for i in range(len(chars) - (self.ngrams - 1))
        ]
        return ngrams

    def _build_word_ngrams(self, text: str) -> List[str]:
        """
        针对不含中文的文本，使用“英文单词 n-gram”。

        处理步骤：
        1. 转为小写；
        2. 去除标点，仅保留“单词字符 + 空白”；
        3. 按空格 split 得到单词序列；
        4. 按连续 n 个“单词”构造 n-gram。
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        text = text.lower()
        # 保留单词字符和空白（英文场景）
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()

        if len(words) < self.ngrams:
            return []

        ngrams = [
            " ".join(words[i : i + self.ngrams])
            for i in range(len(words) - (self.ngrams - 1))
        ]
        return ngrams

    def _compute_repetition_score(self, text: str) -> float:
        """
        根据是否包含中文，选择字符 n-gram 或单词 n-gram，
        并计算 repetition_score = unique_ngrams_count / total_ngrams。
        """
        if self._contains_chinese(text):
            ngrams = self._build_char_ngrams(text)
            mode = "char"
        else:
            ngrams = self._build_word_ngrams(text)
            mode = "word"

        total_ngrams = len(ngrams)
        if total_ngrams == 0:
            # 太短或为空，视为极度重复（得分 0.0）
            self.logger.info(
                f"[ReasoningAnswerNgramFilter] 文本长度不足以构成 n-gram，mode={mode}，视为 repetition_score=0.0"
            )
            return 0.0

        unique_ngrams_count = len(set(ngrams))
        repetition_score = unique_ngrams_count / total_ngrams
        return repetition_score

    # ---------------- DataFrame 校验 ----------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.question_key, self.answer_key]
        forbidden_keys = []  # 这里目前没有真正禁止的列，可以后续扩展

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            self.logger.error(f"[ReasoningAnswerNgramFilter] Missing required column(s): {missing}")
        if conflict:
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] The following column(s) already exist and would be overwritten: {conflict}"
            )

        missing_keys = [key for key in required_keys if key not in dataframe.columns]
        if missing_keys:
            self.logger.error(
                f"[ReasoningAnswerNgramFilter] The following required columns are missing from the dataframe: {missing_keys}"
            )

    # ---------------- 主入口 ----------------
    def run(
        self,
        storage: DataFlowStorage,
        input_question_key: str = "instruction",
        input_answer_key: str = "generated_cot",
    ) -> list:
        self.question_key = input_question_key
        self.answer_key = input_answer_key

        dataframe = storage.read("dataframe")
        self.logger.info(
            f"[ReasoningAnswerNgramFilter] Found {len(dataframe)} rows, columns: {list(dataframe.columns)}"
        )

        self._validate_dataframe(dataframe)

        scores = []
        missing_answer_logged = False

        for _, sample in dataframe.iterrows():
            try:
                q = sample.get(self.question_key, "")
                a = sample.get(self.answer_key, "")
                # question + answer 一起看，防止只在 answer 内截断造成偏差
                answer = f"{q} {a}"
            except Exception:
                if not missing_answer_logged:
                    self.logger.error(
                        "[ReasoningAnswerNgramFilter] 读取 answer 失败，仅使用 question 计算重复度"
                    )
                    missing_answer_logged = True
                answer = sample.get(self.question_key, "")

            score = self._compute_repetition_score(answer)
            scores.append(score)

        scores = np.array(scores, dtype=float)
        indexes = (scores >= self.min_score) & (scores <= self.max_score)

        filtered_df = dataframe[indexes]
        self.logger.info(
            f"[ReasoningAnswerNgramFilter] Filtered down to {len(filtered_df)} rows "
            f"with repetition_score in [{self.min_score}, {self.max_score}]"
        )

        output_file = storage.write(filtered_df)
        self.logger.info(f"[ReasoningAnswerNgramFilter] Results saved to {output_file}")

        # 返回保留的 key 列名
        return [input_question_key, input_answer_key]