"""Deterministic disclaimer injection operator.

Appends a standard risk disclaimer to finance text based on document type
(research note / announcement / marketing material). Pure templating, no LLM.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

DISCLAIMERS = {
    "research": "风险提示：本材料基于公开信息整理，不构成投资建议，据此操作风险自担。",
    "announcement": "声明：本内容仅作信息通报之用，不构成任何形式的投资要约或承诺。",
    "marketing": (
        "重要提示：市场有风险，投资需谨慎。过往业绩不代表未来表现，"
        "本材料不构成收益承诺。"
    ),
}

DEFAULT_DISCLAIMER = "风险提示：本内容不构成投资建议，市场有风险，投资需谨慎。"


@OPERATOR_REGISTRY.register()
class DisclaimerInjector(OperatorABC):
    def __init__(self, custom_disclaimers: dict | None = None):
        super().__init__()
        self.disclaimers = dict(DISCLAIMERS)
        if custom_disclaimers:
            self.disclaimers.update(custom_disclaimers)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "按文档类型（研报/公告/营销材料）为金融文本追加标准风险提示或免责声明，"
                "支持自定义模板，无需 LLM。"
            )
        return (
            "Deterministically append a standard risk disclaimer to finance text "
            "based on document type (research/announcement/marketing); supports "
            "custom templates."
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        input_doc_type: str = "doc_type",
        output_text_with_disclaimer: str = "text_with_disclaimer",
    ):
        import pandas as pd

        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")

        injected = []
        for _, row in dataframe.iterrows():
            text = str(row.get(input_text, "") or "")
            doc_type = str(row.get(input_doc_type, "") or "").strip().lower()
            disclaimer = self.disclaimers.get(doc_type, DEFAULT_DISCLAIMER)
            if not text.strip():
                injected.append("")
            elif disclaimer in text:
                injected.append(text)  # 已含声明则不重复注入
            else:
                injected.append(f"{text.rstrip()}\n\n{disclaimer}")

        result = pd.DataFrame(
            {
                **{c: dataframe[c] for c in dataframe.columns},
                output_text_with_disclaimer: injected,
            }
        )
        storage.write(result)
        return [output_text_with_disclaimer]
