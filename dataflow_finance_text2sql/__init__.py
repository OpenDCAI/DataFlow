"""dataflow_finance_text2sql package.

DataFlow-Extension: 金融领域 Text2SQL（Schema 关联 → SQL 生成 → 语法校验 → SQL 修复）。
导入本包即完成全部算子注册（即插即用）。
"""
from dataflow_finance_text2sql import operator, pipeline  # noqa: F401
from dataflow_finance_text2sql.operator import (  # noqa: F401
    FinanceSchemaLinker,
    FinanceSQLGenerator,
    FinanceSQLSyntaxFilter,
    FinanceSQLRepairRefiner,
)
from dataflow_finance_text2sql.pipeline.finance_text2sql_pipeline import (  # noqa: F401
    FinanceText2SQLPipeline,
)

__version__ = "0.1.0"
__all__ = [
    "FinanceSchemaLinker",
    "FinanceSQLGenerator",
    "FinanceSQLSyntaxFilter",
    "FinanceSQLRepairRefiner",
    "FinanceText2SQLPipeline",
]
