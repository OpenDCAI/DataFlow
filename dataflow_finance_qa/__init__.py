"""dataflow_finance_qa package.

DataFlow-Extension: 金融财报/研报 → 高质量 QA 数据集。
导入本包即完成全部算子注册（即插即用）。
"""
from dataflow_finance_qa import operator, pipeline  # noqa: F401
from dataflow_finance_qa.operator import (  # noqa: F401
    FinancialHallucinationFilter,
    FinancialPDFParser,
    FinancialQAGenerator,
    NumericConsistencyRefiner,
)
from dataflow_finance_qa.pipeline.finance_qa_pipeline import FinanceQAPipeline  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "FinancialPDFParser",
    "FinancialQAGenerator",
    "FinancialHallucinationFilter",
    "NumericConsistencyRefiner",
    "FinanceQAPipeline",
]
