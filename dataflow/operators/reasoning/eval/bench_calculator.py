from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import os
import pandas as pd

@OPERATOR_REGISTRY.register()
class BenchCalculator(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子计算指定 bench 目录下的准确率。\n\n"
                "输入参数：\n"
                "- cache_path: bench 目录路径\n\n"
                "输出参数：\n"
                "- dataframe，包含 total/correct/accuracy"
            )
        elif lang == "en":
            return (
                "This operator calculates accuracy from step1/step2 files "
                "in a specified bench directory.\n\n"
                "Input Parameters:\n"
                "- cache_path: Path to the bench directory\n\n"
                "Output Parameters:\n"
                "- dataframe with total/correct/accuracy"
            )
        else:
            return "BenchAccuracyCalculator computes step1/step2 accuracy for a given path"

    def run(
            self,
            storage: DataFlowStorage,
            cache_path: str = None
    ) -> list:

        path = cache_path or self.cache_path
        if not os.path.isdir(path):
            self.logger.error(f"Invalid path: {path}")
            return []

        files = os.listdir(path)
        step1_files = [f for f in files if "step1" in f]
        step2_files = [f for f in files if "step2" in f]

        if not step1_files or not step2_files:
            self.logger.error(f"Missing step1/step2 files under {path}")
            return []

        step1_file = os.path.join(path, step1_files[0])
        step2_file = os.path.join(path, step2_files[0])

        with open(step1_file, "r", encoding="utf-8") as f1:
            total_count = sum(1 for _ in f1)

        with open(step2_file, "r", encoding="utf-8") as f2:
            correct_count = sum(1 for _ in f2)

        acc = correct_count / total_count if total_count > 0 else 0.0

        df = pd.DataFrame([{
            "bench": os.path.basename(path),
            "total": total_count,
            "correct": correct_count,
            "accuracy": round(acc, 4)
        }])

        output_file = storage.write(df)
        self.logger.info(f"Saved accuracy result to {output_file}")

        return ["dataframe"]
