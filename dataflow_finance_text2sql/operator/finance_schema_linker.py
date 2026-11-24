"""LLM-based finance schema linking operator.

Links a natural-language finance question to the relevant subset of the
database schema (tables / columns / join hints), producing compact context
consumed by the SQL generator.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融数据库 Schema 关联专家。给定自然语言问题与数据库 Schema，"
    "提取回答该问题所需的表、列与连接关系。\n"
    "只输出一个 JSON 对象，格式："
    '{"tables": ["表名"], "columns": ["表.列"], "join_hints": ["连接说明"], "notes": "补充说明"}，'
    "不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance schema-linking expert. Given a natural-language question "
    "and a database schema, extract the tables, columns and join hints needed "
    "to answer it.\n"
    "Output a single JSON object: "
    '{"tables": [...], "columns": [...], "join_hints": [...], "notes": "..."} '
    "and nothing else."
)


@OPERATOR_REGISTRY.register()
class FinanceSchemaLinker(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "将自然语言金融问题关联到数据库 Schema 的相关子集"
                "（表/列/连接提示），输出紧凑上下文供 SQL 生成使用。"
            )
        return (
            "Link a natural-language finance question to the relevant subset of "
            "the database schema (tables/columns/join hints) for SQL generation."
        )

    def _parse_context(self, text: str) -> str:
        import json

        for line in text.splitlines():
            line = line.strip().strip("`")
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    return json.dumps(obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    self.logger.warning("Skip malformed schema-context line")
        return ""

    def run(
        self,
        storage: DataFlowStorage,
        input_question: str = "question",
        input_schema: str = "schema",
        output_schema_context: str = "schema_context",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        contexts = []
        for _, row in dataframe.iterrows():
            question = row[input_question]
            schema = row[input_schema]
            if not question or not str(question).strip():
                contexts.append("")
                continue
            user_input = f"问题：{question}\n\n数据库 Schema：\n{schema}"
            response = self.llm_serving.generate_from_input(
                user_inputs=[user_input],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            contexts.append(self._parse_context(response))

        dataframe[output_schema_context] = contexts
        storage.write(dataframe)
        return [output_schema_context]
