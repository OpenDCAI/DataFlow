"""dataflow_finance_eval package.

DataFlow-Extension: 金融任务评测套件（QA 准确率 / SQL 匹配 / 幻觉检测 / 汇总报告）。
导入本包即完成全部算子注册（即插即用）。
"""
from dataflow_finance_eval import operator, pipeline  # noqa: F401
from dataflow_finance_eval.operator import (  # noqa: F401
    FinanceQASampleEvaluator,
    FinanceSQLSampleEvaluator,
    FinanceHallucinationSampleEvaluator,
    EvalReportGenerator,
)
from dataflow_finance_eval.pipeline.finance_eval_pipeline import FinanceEvalPipeline  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "FinanceQASampleEvaluator",
    "FinanceSQLSampleEvaluator",
    "FinanceHallucinationSampleEvaluator",
    "EvalReportGenerator",
    "FinanceEvalPipeline",
]
