"""Finance compliance pipeline: PII mask → rule filter → risk eval → disclaimer.

PyTorch 风格 Pipeline，遵循 DataFlow 主仓库 Pipeline 约定。
"""
import logging

from dataflow_finance_compliance.operator import (
    ComplianceRiskSampleEvaluator,
    ComplianceRuleFilter,
    DisclaimerInjector,
    FinancialSensitiveInfoMasker,
)


class FinanceCompliancePipeline:
    """端到端金融合规流水线。"""

    def __init__(self, llm_serving, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = kwargs.get("storage")
        self.masker = FinancialSensitiveInfoMasker()
        self.rule_filter = ComplianceRuleFilter(llm_serving=llm_serving)
        self.risk_evaluator = ComplianceRiskSampleEvaluator(llm_serving=llm_serving)
        self.disclaimer_injector = DisclaimerInjector()

    def run(self):
        self.logger.info("FinanceCompliancePipeline start: mask → filter → evaluate → disclaimer")

        storage = self.storage.step()
        self.masker.run(
            storage,
            input_text="text",
            output_masked_text="text_masked",
            output_mask_summary="mask_summary",
        )

        storage = self.storage.step()
        self.rule_filter.run(
            storage, input_text="text_masked", output_violation_reason="violation_reason"
        )

        storage = self.storage.step()
        self.risk_evaluator.run(
            storage,
            input_text="text_masked",
            output_risk_level="risk_level",
            output_risk_reason="risk_reason",
        )

        storage = self.storage.step()
        self.disclaimer_injector.run(
            storage,
            input_text="text_masked",
            input_doc_type="doc_type",
            output_text_with_disclaimer="text_with_disclaimer",
        )

        self.logger.info("FinanceCompliancePipeline done")
