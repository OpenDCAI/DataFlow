"""LLM-based compliance risk sample evaluator.

Scores each finance text row with a risk level (low / medium / high) and a
reason, for downstream audit sampling and reporting.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

SYSTEM_PROMPT_ZH = (
    "你是金融合规风险评估员。对给定金融文本评定风险等级。\n"
    "等级定义：low=常规陈述；medium=表述激进但无明确违规；high=疑似违规或高风险表述。\n"
    '只输出一个 JSON 对象：{"risk_level": "low|medium|high", "reason": "..."}，'
    "不要输出其他内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a finance compliance risk assessor. Rate each text as low / "
    "medium / high risk (high = suspected violation).\n"
    'Output a single JSON object: {"risk_level": "low|medium|high", "reason": '
    '"..."} and nothing else.'
)

VALID_LEVELS = {"low", "medium", "high"}


@OPERATOR_REGISTRY.register()
class ComplianceRiskSampleEvaluator(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "由 LLM 对金融文本逐条评定合规风险等级（low/medium/high）"
                "并给出理由，用于审计抽检与风险报告。"
            )
        return (
            "LLM-based per-sample compliance risk evaluation (low/medium/high) "
            "with reasons, for audit sampling and risk reporting."
        )

    def _parse_risk(self, text: str):
        import json

        for line in text.splitlines():
            line = line.strip().strip("`")
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    level = str(obj.get("risk_level", "")).lower()
                    if level in VALID_LEVELS:
                        return level, str(obj.get("reason", ""))
                except json.JSONDecodeError:
                    self.logger.warning("Skip malformed risk verdict line")
        return "unknown", "parse-failed"

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        output_risk_level: str = "risk_level",
        output_risk_reason: str = "risk_reason",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        levels, reasons = [], []
        for _, row in dataframe.iterrows():
            text = str(row.get(input_text, "") or "")
            if not text.strip():
                levels.append("unknown")
                reasons.append("empty-text")
                continue
            response = self.llm_serving.generate_from_input(
                user_inputs=[text],
                system_prompt=SYSTEM_PROMPT_ZH,
            )[0]
            level, reason = self._parse_risk(response)
            levels.append(level)
            reasons.append(reason)

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_risk_level: levels,
                output_risk_reason: reasons,
            }
        )
        storage.write(result)
        return [output_risk_level, output_risk_reason]
