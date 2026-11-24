"""Deterministic SQL syntax validation filter.

Validates generated SQL with sqlglot when available; falls back to strict
regex/parenthesis checks otherwise. Drops rows whose SQL is not parseable.
"""

import re

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FinanceSQLSyntaxFilter(OperatorABC):
    def __init__(self, dialect: str = "sqlite"):
        super().__init__()
        self.dialect = dialect
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "对生成的金融 SQL 做语法校验（优先 sqlglot 解析，"
                "缺失时退化为正则与括号配平检查），过滤掉不可解析的行并回写错误信息。"
            )
        return (
            "Validate generated finance SQL (sqlglot-based, with regex fallback), "
            "drop unparseable rows and record error messages."
        )

    @staticmethod
    def _fallback_check(sql: str):
        """Return None if OK, else an error message (no sqlglot installed)."""
        if not sql or not sql.strip():
            return "empty SQL"
        if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
            return "statement must start with SELECT/WITH"
        # dangerous keywords
        for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"):
            if re.search(rf"\b{kw}\b", sql, re.IGNORECASE):
                return f"non-read statement keyword: {kw}"
        if sql.count("(") != sql.count(")"):
            return "unbalanced parentheses"
        return None

    def _validate(self, sql: str):
        """Return (is_valid, error_message)."""
        try:
            import sqlglot  # optional dependency
        except ImportError:
            err = self._fallback_check(sql)
            return (err is None, err or "")
        if not sql or not sql.strip():
            return False, "empty SQL"
        try:
            sqlglot.parse_one(sql, read=self.dialect)
            return True, ""
        except Exception as exc:  # noqa: BLE001 - sqlglot raises many types
            return False, str(exc)

    def run(
        self,
        storage: DataFlowStorage,
        input_sql: str = "sql",
        output_sql_error: str = "sql_error",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        kept, kept_errors = [], []
        for _, row in dataframe.iterrows():
            ok, err = self._validate(str(row.get(input_sql, "") or ""))
            if ok:
                record = row.to_dict()
                record[output_sql_error] = ""
                kept.append(record)
            else:
                self.logger.warning(f"Drop row with invalid SQL: {err}")

        result = pd.DataFrame(kept)
        self.logger.info(
            f"SQL syntax filter kept {len(result)}/{len(dataframe)} rows"
        )
        storage.write(result)
        return [output_sql_error]
