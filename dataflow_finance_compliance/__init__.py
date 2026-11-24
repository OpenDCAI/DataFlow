"""dataflow_finance_compliance package.

DataFlow-Extension: 金融合规流水线（敏感信息脱敏 → 合规规则过滤 → 风险评估 → 免责声明注入）。
导入本包即完成全部算子注册（即插即用）。
"""
from dataflow_finance_compliance import operator, pipeline  # noqa: F401
from dataflow_finance_compliance.operator import (  # noqa: F401
    FinancialSensitiveInfoMasker,
    ComplianceRuleFilter,
    ComplianceRiskSampleEvaluator,
    DisclaimerInjector,
)
from dataflow_finance_compliance.pipeline.finance_compliance_pipeline import (  # noqa: F401
    FinanceCompliancePipeline,
)

__version__ = "0.1.0"
__all__ = [
    "FinancialSensitiveInfoMasker",
    "ComplianceRuleFilter",
    "ComplianceRiskSampleEvaluator",
    "DisclaimerInjector",
    "FinanceCompliancePipeline",
]
