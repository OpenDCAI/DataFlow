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
class F1Scorer(OperatorABC):

    def __init__(self, prediction_key, ground_truth_key):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key
        self.logger.info(f"{self.__class__.__name__} initialized.")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "用于评估预测答案与多个参考答案之间的 F1 分数"

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

    def run(self, storage: DataFlowStorage, output_key):
        dataframe = storage.read("dataframe")
        self.output_key = output_key
        self._validate_dataframe(dataframe)
        scores = self.eval(dataframe)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
