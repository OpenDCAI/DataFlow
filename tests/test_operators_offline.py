"""离线单测：通过 stub 替身模拟 DataFlow 框架，验证算子注册与行为约定。

无需安装 open-dataflow / 任何 LLM 服务，CI 可直接运行。
"""
import inspect
import json
import logging
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

from dataflow_finance_qa import (  # noqa: E402
    FinancialHallucinationFilter,
    FinancialPDFParser,
    FinancialQAGenerator,
    NumericConsistencyRefiner,
)
from dataflow.utils.registry import OPERATOR_REGISTRY  # noqa: E402


EXPECTED_OPS = [
    "FinancialPDFParser",
    "FinancialQAGenerator",
    "FinancialHallucinationFilter",
    "NumericConsistencyRefiner",
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
    """按 system_prompt 区分三类调用。"""

    def __init__(self, grounded=True):
        self.grounded = grounded
        self.calls = 0

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls += len(user_inputs)
        out = []
        for _ in user_inputs:
            if "质检员" in system_prompt:  # judge
                out.append(json.dumps({"grounded": self.grounded, "reason": "stub"}))
            elif "修订专家" in system_prompt:  # refiner
                out.append("[refined] " + user_inputs[0].split("答案：")[-1])
            else:  # generator
                out.append(
                    json.dumps({"question": "营收是多少？", "answer": "128.6亿元。"})
                    + "\n"
                    + json.dumps({"question": "毛利润是多少？", "answer": "42.4亿元。"})
                )
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
        {
            "text": "甲公司2025年度营业收入128.6亿元，营业成本86.2亿元。",
        },
        {"text": "乙银行2025年末不良贷款率1.32%。"},
    ]
)


def test_generator_filter_refiner_e2e():
    llm = FakeLLM(grounded=True)
    storage = FakeStorage(BASE_DF.copy())

    FinancialQAGenerator(llm_serving=llm).run(
        storage, input_text="text", output_question="question", output_answer="answer"
    )
    assert storage.df["question"].notna().all()

    FinancialHallucinationFilter(llm_serving=llm).run(
        storage, input_text="text", input_question="question", input_answer="answer"
    )
    assert len(storage.df) == 4  # 每个 chunk 展开 2 条 QA，全部通过

    NumericConsistencyRefiner(llm_serving=llm).run(
        storage, input_text="text", input_answer="answer", output_answer="answer_refined"
    )
    assert storage.df["answer_refined"].str.startswith("[refined]").all()


def test_hallucination_filter_drops_ungrounded():
    llm = FakeLLM(grounded=False)
    storage = FakeStorage(BASE_DF.copy())
    FinancialQAGenerator(llm_serving=FakeLLM()).run(
        storage, input_text="text", output_question="question", output_answer="answer"
    )
    before = len(storage.df)
    FinancialHallucinationFilter(llm_serving=llm).run(
        storage, input_text="text", input_question="question", input_answer="answer"
    )
    assert len(storage.df) == 0 and before > 0


def test_evalset_seed_format():
    path = __file__.rsplit("/", 1)[0].replace("tests", "eval") + "/finance_qa_eval_set.jsonl"
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "eval", "finance_qa_eval_set.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 30
    required = {"id", "qtype", "difficulty", "source_text", "question", "answer"}
    for row in rows:
        assert required.issubset(row.keys()), f"样本 {row.get('id')} 缺字段"
        assert row["qtype"] in {"factual", "numeric", "causal"}
