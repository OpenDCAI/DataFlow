"""Finance Text2SQL pipeline: schema link → SQL generate → syntax filter → repair.

PyTorch 风格 Pipeline，遵循 DataFlow 主仓库 Pipeline 约定。
"""
import logging

from dataflow_finance_text2sql.operator import (
    FinanceSchemaLinker,
    FinanceSQLGenerator,
    FinanceSQLRepairRefiner,
    FinanceSQLSyntaxFilter,
)


class FinanceText2SQLPipeline:
    """端到端金融 Text2SQL 流水线。"""

    def __init__(self, llm_serving, dialect: str = "sqlite", **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = kwargs.get("storage")
        self.linker = FinanceSchemaLinker(llm_serving=llm_serving)
        self.generator = FinanceSQLGenerator(llm_serving=llm_serving, dialect=dialect)
        self.syntax_filter = FinanceSQLSyntaxFilter(dialect=dialect)
        self.repairer = FinanceSQLRepairRefiner(llm_serving=llm_serving, dialect=dialect)

    def run(self):
        self.logger.info("FinanceText2SQLPipeline start: link → generate → filter → repair")

        storage = self.storage.step()
        self.linker.run(
            storage,
            input_question="question",
            input_schema="schema",
            output_schema_context="schema_context",
        )

        storage = self.storage.step()
        self.generator.run(
            storage,
            input_question="question",
            input_schema_context="schema_context",
            output_sql="sql",
        )

        storage = self.storage.step()
        self.syntax_filter.run(storage, input_sql="sql", output_sql_error="sql_error")

        storage = self.storage.step()
        self.repairer.run(
            storage,
            input_sql="sql",
            input_sql_error="sql_error",
            input_schema_context="schema_context",
            output_sql_repaired="sql_repaired",
        )

        self.logger.info("FinanceText2SQLPipeline done")
