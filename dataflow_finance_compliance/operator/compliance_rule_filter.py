"""LLM-based compliance rule filter.

Judges finance text against a set of compliance rules (return promises,
stock-touting inducement, exaggerated claims). Drops rows judged
non-compliant; keeps compliant rows unchanged.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融内容合规质检员。根据合规规则判断给定金融文本是否合规。\n"
    "规则要点：\n"
    "1. 不得含收益承诺或保本暗示（如'稳赚''保本保息''保证翻倍'）；\n"
    "2. 不得含荐股诱导（如'马上买入''私消息我带你操作'）；\n"
    "3. 不得夸大宣传（如'史上最强''稳赢大盘10倍'）。\n"
    '只输出一个 JSON 对象：{"compliant": true/false, "violations": ["规则名"], "reason": "..."}，'
    "不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance compliance reviewer. Judge whether the given finance "
    "text violates rules: return promises, stock-touting inducement, or "
    "exaggerated claims.\n"
    'Output a single JSON object: {"compliant": true/false, "violations": [...], '
    '"reason": "..."} and nothing else.'
)


@OPERATOR_REGISTRY.register()
class ComplianceRuleFilter(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "按合规规则（收益承诺/荐股诱导/夸大宣传）由 LLM 判定金融文本合规性，"
                "过滤不合规行并保留违规原因。"
            )
        return (
            "LLM-based compliance rule filter (return promises / touting / "
            "exaggeration); drops non-compliant rows and keeps violation reasons."
        )

    def _parse_verdict(self, text: str):
        import json

        for line in text.splitlines():
            line = line.strip().strip("`")
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    return bool(obj.get("compliant", True)), str(obj.get("reason", ""))
                except json.JSONDecodeError:
                    self.logger.warning("Skip malformed compliance verdict line")
        # 解析失败按合规处理（宁可保留交人工复核，不误杀）
        return True, "parse-failed-default-keep"

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        output_violation_reason: str = "violation_reason",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        kept = []
        for _, row in dataframe.iterrows():
            text = str(row.get(input_text, "") or "")
            if not text.strip():
                record = row.to_dict()
                record[output_violation_reason] = ""
                kept.append(record)
                continue
            response = self.llm_serving.generate_from_input(
                user_inputs=[text],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            compliant, reason = self._parse_verdict(response)
            if compliant:
                record = row.to_dict()
                record[output_violation_reason] = ""
                kept.append(record)
            else:
                self.logger.warning(f"Drop non-compliant row: {reason}")

        result = pd.DataFrame(kept)
        self.logger.info(
            f"Compliance filter kept {len(result)}/{len(dataframe)} rows"
        )
        storage.write(result)
        return [output_violation_reason]
