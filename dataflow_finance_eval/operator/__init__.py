"""dataflow_finance_eval.operator package.

扫描导入全部算子并注册进 DataFlow 主仓库 registry。
"""
import logging

from typing import TYPE_CHECKING

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:  # pragma: no cover - LazyLoader 提示用
    from .finance_qa_sample_evaluator import FinanceQASampleEvaluator
    from .finance_sql_sample_evaluator import FinanceSQLSampleEvaluator
    from .finance_hallucination_sample_evaluator import FinanceHallucinationSampleEvaluator
    from .eval_report_generator import EvalReportGenerator

# 运行时导入触发 @OPERATOR_REGISTRY.register()
from .finance_qa_sample_evaluator import FinanceQASampleEvaluator  # noqa: E402,F401
from .finance_sql_sample_evaluator import FinanceSQLSampleEvaluator  # noqa: E402,F401
from .finance_hallucination_sample_evaluator import FinanceHallucinationSampleEvaluator  # noqa: E402,F401
from .eval_report_generator import EvalReportGenerator  # noqa: E402,F401

__all__ = [
    "FinanceQASampleEvaluator",
    "FinanceSQLSampleEvaluator",
    "FinanceHallucinationSampleEvaluator",
    "EvalReportGenerator",
]
