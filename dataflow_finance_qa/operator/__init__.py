"""dataflow_finance_qa.operator package.

扫描导入全部算子并注册进 DataFlow 主仓库 registry。
"""
import logging

from typing import TYPE_CHECKING

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:  # pragma: no cover - LazyLoader 提示用
    from .financial_pdf_parser import FinancialPDFParser
    from .financial_qa_generator import FinancialQAGenerator
    from .financial_hallucination_filter import FinancialHallucinationFilter
    from .numeric_consistency_refiner import NumericConsistencyRefiner

# 运行时导入触发 @OPERATOR_REGISTRY.register()
from .financial_pdf_parser import FinancialPDFParser  # noqa: E402,F401
from .financial_qa_generator import FinancialQAGenerator  # noqa: E402,F401
from .financial_hallucination_filter import FinancialHallucinationFilter  # noqa: E402,F401
from .numeric_consistency_refiner import NumericConsistencyRefiner  # noqa: E402,F401

__all__ = [
    "FinancialPDFParser",
    "FinancialQAGenerator",
    "FinancialHallucinationFilter",
    "NumericConsistencyRefiner",
]
