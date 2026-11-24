"""Deterministic finance SQL sample evaluator.

Compares a generated SQL with the gold SQL: normalized text match by
default; optionally uses sqlglot to compare parsed ASTs (dialect-agnostic).
"""

import re

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FinanceSQLSampleEvaluator(OperatorABC):
    def __init__(self, use_ast: bool = False, dialect: str = "sqlite"):
        super().__init__()
        self.use_ast = use_ast  # True 时用 sqlglot AST 比较（需安装 sqlglot）
        self.dialect = dialect
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "确定性评测金融 SQL：默认归一化文本匹配；安装 sqlglot 后可按 "
                "AST 比较（与空白/别名无关），无需 LLM。"
            )
        return (
            "Deterministic finance SQL evaluation: normalized text match by "
            "default; optional sqlglot AST comparison when installed; no LLM."
        )

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        sql = str(sql or "").lower()
        sql = re.sub(r"\s+", " ", sql).strip().rstrip(";")
        sql = re.sub(r"\s*,\s*", ", ", sql)
        sql = re.sub(r"\s*=\s*", " = ", sql)
        return sql

    def _score_ast(self, sql: str, gold: str) -> float:
        import sqlglot  # optional dependency
        import sqlglot.expressions as exp

        try:
            a = sqlglot.parse_one(sql, read=self.dialect)
            b = sqlglot.parse_one(gold, read=self.dialect)
        except Exception:  # noqa: BLE001 - 解析失败记 0 分
            return 0.0
        if a.sql(dialect=self.dialect) == b.sql(dialect=self.dialect):
            return 1.0
        # 列名集合与表名集合比较（放宽别名差异）
        cols_a = {c.name for c in a.find_all(exp.Column)}
        cols_b = {c.name for c in b.find_all(exp.Column)}
        tabs_a = {t.name for t in a.find_all(exp.Table)}
        tabs_b = {t.name for t in b.find_all(exp.Table)}
        if cols_a == cols_b and tabs_a == tabs_b:
            return 0.5
        return 0.0

    def run(
        self,
        storage: DataFlowStorage,
        input_sql: str = "sql",
        input_gold_sql: str = "gold_sql",
        output_sql_score: str = "sql_score",
        output_sql_correct: str = "sql_correct",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        scores = []
        for _, row in dataframe.iterrows():
            sql = str(row.get(input_sql, "") or "")
            gold = str(row.get(input_gold_sql, "") or "")
            if not sql.strip() or not gold.strip():
                scores.append(0.0)
            elif self.use_ast:
                scores.append(self._score_ast(sql, gold))
            else:
                scores.append(1.0 if self._normalize_sql(sql) == self._normalize_sql(gold) else 0.0)

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_sql_score: scores,
                output_sql_correct: [s >= 1.0 for s in scores],
            }
        )
        storage.write(result)
        return [output_sql_score, output_sql_correct]
