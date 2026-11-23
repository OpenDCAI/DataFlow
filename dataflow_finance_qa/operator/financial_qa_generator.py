"""LLM-based financial QA generation operator.

Generates question-answer pairs from financial text chunks
(annual report excerpts, research notes, announcements),
constrained to be grounded in the source text.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融数据标注专家。基于给定的金融文本（财报/研报/公告片段）生成问答对。\n"
    "要求：\n"
    "1. 问题必须能且只能从给定文本中回答（杜绝外部知识编造）；\n"
    "2. 覆盖多种类型：数值计算、事实提取、因果/驱动因素分析；\n"
    "3. 答案保留原文中的精确数字与单位；\n"
    "4. 每行输出一个 JSON 对象，格式：{\"question\": ..., \"answer\": ...}，不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a financial data annotation expert. Generate QA pairs from the "
    "given financial text (report / research note / announcement excerpt).\n"
    "Rules:\n"
    "1. Each question must be answerable strictly from the given text;\n"
    "2. Cover multiple types: numeric reasoning, fact extraction, causal analysis;\n"
    "3. Keep exact numbers and units from the source;\n"
    '4. Output one JSON object per line: {"question": ..., "answer": ...}, nothing else.'
)


@OPERATOR_REGISTRY.register()
class FinancialQAGenerator(OperatorABC):
    def __init__(self, llm_serving, num_questions_per_chunk: int = 3):
        super().__init__()
        self.llm_serving = llm_serving
        self.num_questions_per_chunk = num_questions_per_chunk
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于金融文本片段生成严格锚定原文的问答对，"
                "覆盖数值计算、事实提取与归因分析三类问题。"
            )
        return (
            "Generate source-grounded QA pairs from financial text chunks, "
            "covering numeric reasoning, fact extraction and causal analysis."
        )

    def _parse_jsonl(self, text: str):
        import json

        pairs = []
        for line in text.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if obj.get("question") and obj.get("answer"):
                    pairs.append((str(obj["question"]), str(obj["answer"])))
            except json.JSONDecodeError:
                self.logger.warning("Skip malformed QA line")
        return pairs

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        output_question: str = "question",
        output_answer: str = "answer",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")
        records = []
        for _, row in dataframe.iterrows():
            text = row[input_text]
            base = row.to_dict()
            if not text or not text.strip():
                base[output_question] = None
                base[output_answer] = None
                records.append(base)
                continue
            user_input = (
                f"请基于以下文本生成 {self.num_questions_per_chunk} 个问答对：\n\n{text}"
            )
            response = self.llm_serving.generate_from_input(
                user_inputs=[user_input],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            pairs = self._parse_jsonl(response)
            if pairs:
                for question, answer in pairs:
                    records.append(
                        {**base, output_question: question, output_answer: answer}
                    )
            else:
                base[output_question] = None
                base[output_answer] = None
                records.append(base)
        result = pd.DataFrame(records)
        self.logger.info(f"Generated {len(result)} QA rows from {len(dataframe)} chunks")
        storage.write(result)
        return [output_question, output_answer]
