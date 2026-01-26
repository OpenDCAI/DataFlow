import re
import string
from collections import Counter
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger


@OPERATOR_REGISTRY.register()
class AgenticRAGQAF1SampleEvaluator(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于计算预测答案与参考答案（Ground Truth）之间的 F1 分数。它通过词袋模型（Token-based）评估文本相似度，支持答案归一化及多参考答案匹配。\n\n"
                "输入参数：\n"
                "- input_prediction_key: 预测答案字段名（默认值：\"refined_answer\"）\n"
                "- input_ground_truth_key: 参考答案字段名，支持字符串或列表（默认值：\"golden_doc_answer\"）\n"
                "- output_key: 输出 F1 分数字段名（默认值：\"F1Score\"）\n"
            )
        elif lang == "en":
            return (
                "This operator computes the F1 score between predicted answers and reference (ground truth) answers. "
                "It evaluates text similarity based on tokens, supports answer normalization, and matches against multiple reference answers.\n\n"
                "Input Parameters:\n"
                "- input_prediction_key: Field name for the predicted answer (default: \"refined_answer\")\n"
                "- input_ground_truth_key: Field name for the ground truth answer, supports string or list (default: \"golden_doc_answer\")\n"
                "- output_key: Field name for the output F1 score (default: \"F1Score\")\n"
            )
        else:
            return "AgenticRAGQAF1SampleEvaluator computes F1 scores for QA evaluation based on token overlap."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.prediction_key, self.ground_truth_key]
        forbidden_keys = [self.output_key ]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(self, prediction: str, ground_truths) -> float:
        if prediction is None or ground_truths is None:
            return 0.0

        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        max_f1 = 0.0

        for ground_truth in ground_truths:
            if ground_truth is None:
                continue

            normalized_prediction = self.normalize_answer(prediction)
            normalized_ground_truth = self.normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] or normalized_ground_truth in ["yes", "no", "noanswer"]:
                if normalized_prediction != normalized_ground_truth:
                    continue

            pred_tokens = normalized_prediction.split()
            gold_tokens = normalized_ground_truth.split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

        return max_f1

    def eval(self, dataframe: pd.DataFrame) -> list:
        self.logger.info(f"Evaluating {self.output_key}...")
        f1_scores = []

        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="F1Scorer Evaluating..."):
            prediction = row.get(self.prediction_key, None)
            ground_truths = row.get(self.ground_truth_key, None)
            score = self.compute_f1(prediction, ground_truths)
            f1_scores.append(score)

        self.logger.info("Evaluation complete!")
        return f1_scores

    def run(self, 
            storage: DataFlowStorage, 
            input_prediction_key:str ="refined_answer",
            input_ground_truth_key:str ="golden_doc_answer",
            output_key:str ="F1Score",
            ):
        dataframe = storage.read("dataframe")
        self.output_key = output_key
        self.prediction_key = input_prediction_key
        self.ground_truth_key = input_ground_truth_key
        self._validate_dataframe(dataframe)
        scores = self.eval(dataframe)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
