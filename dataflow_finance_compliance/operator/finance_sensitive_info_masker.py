"""Deterministic PII / sensitive-info masking operator for finance text.

Masks Chinese ID numbers, mainland mobile numbers, bank card numbers and
email addresses with type-tagged placeholders, and records a per-row mask
summary. Pure regex, no LLM required.
"""

import json
import re

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

PATTERNS = {
    "id_card": (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证]"),
    "mobile": (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    "bank_card": (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[银行卡]"),
    "email": (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[邮箱]"),
}

# 匹配顺序很重要：先身份证（18位），再银行卡（16-19位），避免手机号/卡号互相吞
MASK_ORDER = ["id_card", "bank_card", "mobile", "email"]


@OPERATOR_REGISTRY.register()
class FinancialSensitiveInfoMasker(OperatorABC):
    def __init__(self, extra_patterns: dict | None = None):
        super().__init__()
        self.patterns = dict(PATTERNS)
        if extra_patterns:
            for name, (regex, placeholder) in extra_patterns.items():
                self.patterns[name] = (re.compile(regex), placeholder)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "对金融文本中的敏感信息（身份证/手机号/银行卡/邮箱）做确定性脱敏，"
                "输出脱敏后文本与逐类计数摘要，无需 LLM。"
            )
        return (
            "Deterministically mask sensitive info (ID card / mobile / bank card / "
            "email) in finance text, outputting masked text and a mask summary."
        )

    def mask_text(self, text: str):
        """Return (masked_text, counts_dict)."""
        counts = {}
        masked = text or ""
        for name in MASK_ORDER:
            if name not in self.patterns:
                continue
            regex, placeholder = self.patterns[name]
            masked, n = regex.subn(placeholder, masked)
            if n:
                counts[name] = n
        return masked, counts

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        output_masked_text: str = "text_masked",
        output_mask_summary: str = "mask_summary",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        masked_texts, summaries = [], []
        total = 0
        for _, row in dataframe.iterrows():
            masked, counts = self.mask_text(str(row.get(input_text, "") or ""))
            masked_texts.append(masked)
            summaries.append(json.dumps(counts, ensure_ascii=False))
            total += sum(counts.values())

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_masked_text: masked_texts,
                output_mask_summary: summaries,
            }
        )
        self.logger.info(f"Masked {total} sensitive items in {len(dataframe)} rows")
        storage.write(result)
        return [output_masked_text, output_mask_summary]
