"""LLM-based finance SQL generation operator.

Generates a single SQL statement per finance question, grounded in the
schema context produced by FinanceSchemaLinker.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融 SQL 生成专家。根据问题与 Schema 关联上下文，生成一条可执行的 SQL。\n"
    "要求：\n"
    "1. 只使用上下文与 Schema 中出现的表和列，不得虚构；\n"
    "2. 金额/比率字段注意单位与精度，比率计算注意除零保护（NULLIF）；\n"
    "3. 只输出一个 ```sql 代码块，块内仅一条 SQL，不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance SQL expert. Generate ONE executable SQL statement from "
    "the question and schema context.\n"
    "Rules:\n"
    "1. Use only tables/columns present in the schema context;\n"
    "2. Guard ratio computations against division by zero (NULLIF);\n"
    "3. Output a single ```sql code block containing one statement, nothing else."
)


@OPERATOR_REGISTRY.register()
class FinanceSQLGenerator(OperatorABC):
    def __init__(self, llm_serving, dialect: str = "sqlite"):
        super().__init__()
        self.llm_serving = llm_serving
        self.dialect = dialect
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于金融问题与 Schema 关联上下文生成单条 SQL，"
                "严格限定在已知表列内，并对比率计算做除零保护。"
            )
        return (
            "Generate a single SQL statement from a finance question and schema "
            "context, restricted to known tables/columns with NULLIF guards."
        )

    def _extract_sql(self, text: str) -> str:
        if "```sql" in text:
            body = text.split("```sql", 1)[1]
            body = body.split("```", 1)[0]
            return body.strip().rstrip(";").strip()
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith(("SELECT", "WITH")):
                return line.rstrip(";").strip()
        return ""

    def run(
        self,
        storage: DataFlowStorage,
        input_question: str = "question",
        input_schema_context: str = "schema_context",
        output_sql: str = "sql",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        sqls = []
        for _, row in dataframe.iterrows():
            question = row[input_question]
            context = row.get(input_schema_context, "")
            if not question or not str(question).strip():
                sqls.append("")
                continue
            user_input = (
                f"目标方言：{self.dialect}\n问题：{question}\n"
                f"Schema 关联上下文：{context}"
            )
            response = self.llm_serving.generate_from_input(
                user_inputs=[user_input],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            sqls.append(self._extract_sql(response))

        dataframe[output_sql] = sqls
        storage.write(dataframe)
        return [output_sql]
