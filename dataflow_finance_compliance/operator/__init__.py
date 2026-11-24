"""dataflow_finance_compliance.operator package.

扫描导入全部算子并注册进 DataFlow 主仓库 registry。
"""
import logging

from typing import TYPE_CHECKING

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:  # pragma: no cover - LazyLoader 提示用
    from .finance_sensitive_info_masker import FinancialSensitiveInfoMasker
    from .compliance_rule_filter import ComplianceRuleFilter
    from .compliance_risk_sample_evaluator import ComplianceRiskSampleEvaluator
    from .disclaimer_injector import DisclaimerInjector

# 运行时导入触发 @OPERATOR_REGISTRY.register()
from .finance_sensitive_info_masker import FinancialSensitiveInfoMasker  # noqa: E402,F401
from .compliance_rule_filter import ComplianceRuleFilter  # noqa: E402,F401
from .compliance_risk_sample_evaluator import ComplianceRiskSampleEvaluator  # noqa: E402,F401
from .disclaimer_injector import DisclaimerInjector  # noqa: E402,F401

__all__ = [
    "FinancialSensitiveInfoMasker",
    "ComplianceRuleFilter",
    "ComplianceRiskSampleEvaluator",
    "DisclaimerInjector",
]
