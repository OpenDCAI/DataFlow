"""Deterministic aggregate report generator.

Aggregates per-sample scores into a one-row summary DataFrame:
accuracy / hallucination rate / sample count. Terminal operator of the
finance eval pipeline.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class EvalReportGenerator(OperatorABC):
    def __init__(self):
        super().__init__()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "汇总评测明细：聚合 QA 准确率、SQL 准确率与幻觉率，"
                "输出单行汇总 DataFrame，作为评测流水线的终算子。"
            )
        return (
            "Aggregate eval details into a one-row summary (QA accuracy, SQL "
            "accuracy, hallucination rate); terminal operator of the eval pipeline."
        )

    @staticmethod
    def _acc(series) -> float:
        vals = [bool(v) for v in series if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def run(
        self,
        storage: DataFlowStorage,
        input_qa_correct: str = "qa_correct",
        input_sql_correct: str = "sql_correct",
        input_hallucination_flag: str = "hallucination",
        output_summary: str = "eval_summary",
    ):
        import json

        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        report = {
            "sample_count": int(len(dataframe)),
            "qa_accuracy": self._acc(dataframe.get(input_qa_correct)),
            "sql_accuracy": self._acc(dataframe.get(input_sql_correct)),
            "hallucination_rate": None,
        }
        flags = [bool(v) for v in dataframe.get(input_hallucination_flag) if v is not None]
        if flags:
            report["hallucination_rate"] = round(sum(flags) / len(flags), 4)

        result = pd.DataFrame([{output_summary: json.dumps(report, ensure_ascii=False)}])
        self.logger.info(f"Eval report: {report}")
        storage.write(result)
        return [output_summary]
