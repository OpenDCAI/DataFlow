"""离线单测：通过 stub 替身模拟 DataFlow 框架，验证算子注册与行为约定。

无需安装 open-dataflow / 任何 LLM 服务，CI 可直接运行。
"""
import inspect
import json
import logging
import os
import sys
import types

import pandas as pd
import pytest


# ---------------- DataFlow 框架 stub ----------------
def _install_dataflow_stubs():
    if "dataflow" in sys.modules:
        return

    dataflow = types.ModuleType("dataflow")
    core = types.ModuleType("dataflow.core")

    class OperatorABC:
        def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)

    core.OperatorABC = OperatorABC

    utils = types.ModuleType("dataflow.utils")
    registry = types.ModuleType("dataflow.utils.registry")

    class _Registry:
        _obj_map = {}

        @classmethod
        def register(cls):
            def decorator(cls_):
                cls._obj_map[cls_.__name__] = cls_
                return cls_

            return decorator

    registry.OPERATOR_REGISTRY = _Registry

    storage_mod = types.ModuleType("dataflow.utils.storage")

    class DataFlowStorage:  # noqa: D401 - 仅作类型占位
        pass

    storage_mod.DataFlowStorage = DataFlowStorage

    serving = types.ModuleType("dataflow.serving")

    class APILLMServing_request:  # noqa: N801
        pass

    serving.APILLMServing_request = APILLMServing_request

    sys.modules.update(
        {
            "dataflow": dataflow,
            "dataflow.core": core,
            "dataflow.utils": utils,
            "dataflow.utils.registry": registry,
            "dataflow.utils.storage": storage_mod,
            "dataflow.serving": serving,
        }
    )


_install_dataflow_stubs()

from dataflow_finance_compliance import (  # noqa: E402
    ComplianceRiskSampleEvaluator,
    ComplianceRuleFilter,
    DisclaimerInjector,
    FinancialSensitiveInfoMasker,
)
from dataflow.utils.registry import OPERATOR_REGISTRY  # noqa: E402


EXPECTED_OPS = [
    "FinancialSensitiveInfoMasker",
    "ComplianceRuleFilter",
    "ComplianceRiskSampleEvaluator",
    "DisclaimerInjector",
]


def test_operators_registered():
    for name in EXPECTED_OPS:
        assert name in OPERATOR_REGISTRY._obj_map, f"{name} 未注册进 registry"


@pytest.mark.parametrize("name", EXPECTED_OPS)
def test_get_desc_bilingual(name):
    cls = OPERATOR_REGISTRY._obj_map[name]
    for lang in ("zh", "en"):
        desc = cls.get_desc(lang)
        assert isinstance(desc, str) and len(desc) > 10


@pytest.mark.parametrize("name", EXPECTED_OPS)
def test_run_signature_conventions(name):
    """run() 首参为 storage，其余参数以 input_/output_ 前缀命名（Pipeline 编译依赖）。"""
    sig = inspect.signature(OPERATOR_REGISTRY._obj_map[name].run)
    params = [p for p in sig.parameters if p != "self"]
    assert params[0] == "storage"
    for p in params[1:]:
        assert p.startswith(("input_", "output_")), f"{name}.run 参数 {p} 不符合命名约定"


class FakeLLM:
    """按 system_prompt 关键词区分两类调用：合规质检员 / 风险评估员。"""

    def __init__(self, compliant=True, risk="low"):
        self.compliant = compliant
        self.risk = risk
        self.calls = 0

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls += len(user_inputs)
        out = []
        for _ in user_inputs:
            if "合规质检员" in system_prompt:
                out.append(json.dumps({"compliant": self.compliant, "violations": [], "reason": "stub"}))
            elif "风险评估员" in system_prompt:
                out.append(json.dumps({"risk_level": self.risk, "reason": "stub"}))
            else:
                out.append("{}")
        return out


class FakeStorage:
    def __init__(self, df):
        self.df = df

    def read(self, _key):
        return self.df

    def write(self, df):
        self.df = df
        return df

    def step(self):
        return self


BASE_DF = pd.DataFrame(
    [
        {"text": "客户张三手机号13812345678，邮箱 a@b.com", "doc_type": "research"},
        {"text": "该股下月保证翻倍，稳赚不赔。", "doc_type": "marketing"},
    ]
)


def test_masker_masks_all_pii_types():
    storage = FakeStorage(BASE_DF[["text"]].copy())
    FinancialSensitiveInfoMasker().run(
        storage,
        input_text="text",
        output_masked_text="text_masked",
        output_mask_summary="mask_summary",
    )
    masked0 = storage.df["text_masked"].iloc[0]
    assert "[手机号]" in masked0 and "[邮箱]" in masked0 and "13812345678" not in masked0
    counts = json.loads(storage.df["mask_summary"].iloc[0])
    assert counts.get("mobile") == 1 and counts.get("email") == 1


def test_masker_order_id_card_before_bank_card():
    """18位身份证必须按身份证掩码，不被卡号规则吞掉。"""
    storage = FakeStorage(pd.DataFrame([{"text": "证件号110101199001011234"}]))
    FinancialSensitiveInfoMasker().run(storage, input_text="text", output_masked_text="m", output_mask_summary="s")
    assert storage.df["m"].iloc[0] == "证件号[身份证]"


def test_rule_filter_drops_non_compliant():
    llm = FakeLLM(compliant=False)
    storage = FakeStorage(BASE_DF.copy())
    ComplianceRuleFilter(llm_serving=llm).run(storage, input_text="text", output_violation_reason="violation_reason")
    assert len(storage.df) == 0


def test_rule_filter_keeps_compliant():
    llm = FakeLLM(compliant=True)
    storage = FakeStorage(BASE_DF.copy())
    ComplianceRuleFilter(llm_serving=llm).run(storage, input_text="text", output_violation_reason="violation_reason")
    assert len(storage.df) == 2


def test_risk_evaluator_levels():
    llm = FakeLLM(risk="high")
    storage = FakeStorage(BASE_DF.copy())
    ComplianceRiskSampleEvaluator(llm_serving=llm).run(
        storage, input_text="text", output_risk_level="risk_level", output_risk_reason="risk_reason"
    )
    assert (storage.df["risk_level"] == "high").all()


def test_disclaimer_injection_dedup_and_type():
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"text": "研报正文。", "doc_type": "research"},
                {"text": "已有声明。\n\n风险提示：本材料基于公开信息整理，不构成投资建议，据此操作风险自担。", "doc_type": "research"},
                {"text": "", "doc_type": "research"},
            ]
        )
    )
    DisclaimerInjector().run(
        storage, input_text="text", input_doc_type="doc_type", output_text_with_disclaimer="text_with_disclaimer"
    )
    first = storage.df["text_with_disclaimer"].iloc[0]
    assert "风险提示" in first and first.startswith("研报正文。")
    assert storage.df["text_with_disclaimer"].iloc[1].count("风险提示") == 1  # 不重复注入
    assert storage.df["text_with_disclaimer"].iloc[2] == ""


def test_masker_disclaimer_no_llm_needed():
    """两个确定性算子在 llm_serving=None 时也应可用。"""
    storage = FakeStorage(pd.DataFrame([{"text": "手机13912345678", "doc_type": "research"}]))
    FinancialSensitiveInfoMasker().run(storage, input_text="text", output_masked_text="m", output_mask_summary="s")
    DisclaimerInjector().run(storage, input_text="m", input_doc_type="doc_type", output_text_with_disclaimer="d")
    assert "[手机号]" in storage.df["d"].iloc[0]


def test_evalset_seed_format():
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "finance_compliance_eval_set.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 30
    required = {"id", "doc_type", "expected_compliant", "expected_risk", "contains_sensitive", "text"}
    for row in rows:
        assert required.issubset(row.keys()), f"样本 {row.get('id')} 缺字段"
        assert row["doc_type"] in {"research", "announcement", "marketing"}
        assert row["expected_risk"] in {"low", "medium", "high"}
        assert isinstance(row["expected_compliant"], bool)
