"""LLM-based SQL repair refiner operator.

Takes rows whose SQL failed validation and produces a repaired statement
grounded in the same schema context.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融 SQL 修复专家。给定 SQL、报错信息与 Schema 上下文，"
    "输出修复后的单条 SQL。\n"
    "要求：只使用上下文中的表和列；只输出一个 ```sql 代码块，不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance SQL repair expert. Given a SQL statement, its error and "
    "schema context, output the repaired SQL.\n"
    "Rules: use only tables/columns from the context; output a single ```sql "
    "code block, nothing else."
)


@OPERATOR_REGISTRY.register()
class FinanceSQLRepairRefiner(OperatorABC):
    def __init__(self, llm_serving, dialect: str = "sqlite"):
        super().__init__()
        self.llm_serving = llm_serving
        self.dialect = dialect
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "对语法校验失败的金融 SQL，结合报错信息与 Schema 上下文"
                "生成修复后的 SQL。"
            )
        return (
            "Repair finance SQL that failed syntax validation, using the error "
            "message and schema context to produce a fixed statement."
        )

    def _extract_sql(self, text: str) -> str:
        if "```sql" in text:
            body = text.split("```sql", 1)[1].split("```", 1)[0]
            return body.strip().rstrip(";").strip()
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith(("SELECT", "WITH")):
                return line.rstrip(";").strip()
        return ""

    def run(
        self,
        storage: DataFlowStorage,
        input_sql: str = "sql",
        input_sql_error: str = "sql_error",
        input_schema_context: str = "schema_context",
        output_sql_repaired: str = "sql_repaired",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        repaired = []
        for _, row in dataframe.iterrows():
            sql = str(row.get(input_sql, "") or "")
            error = str(row.get(input_sql_error, "") or "")
            context = row.get(input_schema_context, "")
            if error and error.strip():
                user_input = (
                    f"方言：{self.dialect}\n待修复 SQL：\n{sql}\n"
                    f"报错信息：{error}\nSchema 上下文：{context}"
                )
                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_input],
                    system_prompt=SYSTEM_PROMPT_ZH,
                )[0]
                repaired.append(self._extract_sql(response))
            else:
                repaired.append(sql)  # 无报错则原样保留

        dataframe[output_sql_repaired] = repaired
        storage.write(dataframe)
        return [output_sql_repaired]
