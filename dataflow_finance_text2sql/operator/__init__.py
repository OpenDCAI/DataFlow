"""dataflow_finance_text2sql.operator package.

扫描导入全部算子并注册进 DataFlow 主仓库 registry。
"""
import logging

from typing import TYPE_CHECKING

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:  # pragma: no cover - LazyLoader 提示用
    from .finance_schema_linker import FinanceSchemaLinker
    from .finance_sql_generator import FinanceSQLGenerator
    from .finance_sql_syntax_filter import FinanceSQLSyntaxFilter
    from .finance_sql_repair_refiner import FinanceSQLRepairRefiner

# 运行时导入触发 @OPERATOR_REGISTRY.register()
from .finance_schema_linker import FinanceSchemaLinker  # noqa: E402,F401
from .finance_sql_generator import FinanceSQLGenerator  # noqa: E402,F401
from .finance_sql_syntax_filter import FinanceSQLSyntaxFilter  # noqa: E402,F401
from .finance_sql_repair_refiner import FinanceSQLRepairRefiner  # noqa: E402,F401

__all__ = [
    "FinanceSchemaLinker",
    "FinanceSQLGenerator",
    "FinanceSQLSyntaxFilter",
    "FinanceSQLRepairRefiner",
]
