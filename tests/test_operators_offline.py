"""离线单测：通过 stub 替身模拟 DataFlow 框架，验证算子注册与行为约定。

无需安装 open-dataflow / sqlglot / 任何 LLM 服务，CI 可直接运行。
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

from dataflow_finance_eval import (  # noqa: E402
    EvalReportGenerator,
    FinanceHallucinationSampleEvaluator,
    FinanceQASampleEvaluator,
    FinanceSQLSampleEvaluator,
)
from dataflow.utils.registry import OPERATOR_REGISTRY  # noqa: E402


EXPECTED_OPS = [
    "FinanceQASampleEvaluator",
    "FinanceSQLSampleEvaluator",
    "FinanceHallucinationSampleEvaluator",
    "EvalReportGenerator",
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
    """幻觉检测员调用：grounded 可控。"""

    def __init__(self, grounded=True):
        self.grounded = grounded
        self.calls = 0

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls += len(user_inputs)
        assert "幻觉检测员" in system_prompt
        return [json.dumps({"grounded": self.grounded, "reason": "stub"}) for _ in user_inputs]


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


def test_qa_evaluator_exact_numeric_and_fuzzy():
    ev = FinanceQASampleEvaluator()
    assert ev._score_one("128.6亿元", "128.6亿元") == 1.0
    assert ev._score_one("128.6 亿元。", "128.6亿元") == 1.0  # 归一化
    assert ev._score_one("同比增长15%", "15%") == 1.0  # 数值容差匹配
    assert ev._score_one("130亿元", "128.6亿元") == 0.0  # 数值超容差
    assert ev._score_one("经营现金流净额为正且同比改善", "为正且同比改善") == 1.0  # 字符重叠
    assert ev._score_one("", "x") == 0.0


def test_qa_evaluator_run():
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"answer": "128.6亿元", "gold_answer": "128.6亿元"},
                {"answer": "130亿元", "gold_answer": "128.6亿元"},
            ]
        )
    )
    FinanceQASampleEvaluator().run(
        storage, input_answer="answer", input_gold_answer="gold_answer",
        output_qa_score="qa_score", output_qa_correct="qa_correct",
    )
    assert storage.df["qa_correct"].tolist() == [True, False]


def test_sql_evaluator_normalized_match():
    ev = FinanceSQLSampleEvaluator()
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"sql": "select  company,revenue  from revenue;", "gold_sql": "SELECT company, revenue FROM revenue"},
                {"sql": "SELECT cost FROM cost", "gold_sql": "SELECT revenue FROM revenue"},
            ]
        )
    )
    FinanceSQLSampleEvaluator().run(
        storage, input_sql="sql", input_gold_sql="gold_sql",
        output_sql_score="sql_score", output_sql_correct="sql_correct",
    )
    assert storage.df["sql_correct"].tolist() == [True, False]


def test_hallucination_evaluator_flags_ungrounded():
    llm = FakeLLM(grounded=False)
    storage = FakeStorage(pd.DataFrame([{"text": "原文", "question": "q", "answer": "a"}]))
    FinanceHallucinationSampleEvaluator(llm_serving=llm).run(
        storage, input_text="text", input_question="question", input_answer="answer",
        output_hallucination_flag="hallucination", output_hallucination_reason="hallucination_reason",
    )
    assert storage.df["hallucination"].iloc[0] is True or storage.df["hallucination"].iloc[0] == True  # noqa: E712
    # 空答案默认判幻觉
    storage2 = FakeStorage(pd.DataFrame([{"text": "原文", "question": "q", "answer": ""}]))
    FinanceHallucinationSampleEvaluator(llm_serving=FakeLLM()).run(
        storage2, input_text="text", input_question="question", input_answer="answer",
        output_hallucination_flag="hallucination", output_hallucination_reason="hallucination_reason",
    )
    assert bool(storage2.df["hallucination"].iloc[0])


def test_report_generator_aggregates():
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"qa_correct": True, "sql_correct": True, "hallucination": False},
                {"qa_correct": False, "sql_correct": True, "hallucination": True},
            ]
        )
    )
    EvalReportGenerator().run(
        storage, input_qa_correct="qa_correct", input_sql_correct="sql_correct",
        input_hallucination_flag="hallucination", output_summary="eval_summary",
    )
    report = json.loads(storage.df["eval_summary"].iloc[0])
    assert report["sample_count"] == 2
    assert report["qa_accuracy"] == 0.5
    assert report["sql_accuracy"] == 1.0
    assert report["hallucination_rate"] == 0.5


def test_sample_set_matches_evaluator_expectations():
    """抽样校验：金标 expected_* 与确定性评测逻辑对拍。"""
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "finance_eval_sample_set.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 30
    required = {"id", "difficulty", "text", "question", "answer", "gold_answer",
                "sql", "gold_sql", "gold_grounded", "expected_qa_correct", "expected_sql_correct"}
    for row in rows:
        assert required.issubset(row.keys()), f"样本 {row.get('id')} 缺字段"

    qa_ev = FinanceQASampleEvaluator()
    sql_ev = FinanceSQLSampleEvaluator()
    mismatch_qa, mismatch_sql = [], []
    for row in rows:
        qa_ok = qa_ev._score_one(row["answer"], row["gold_answer"]) >= 1.0
        sql_ok = sql_ev._normalize_sql(row["sql"]) == sql_ev._normalize_sql(row["gold_sql"])
        if qa_ok != row["expected_qa_correct"]:
            mismatch_qa.append(row["id"])
        if sql_ok != row["expected_sql_correct"]:
            mismatch_sql.append(row["id"])
    assert not mismatch_qa, f"金标 qa_correct 与确定性逻辑不一致: {mismatch_qa}"
    assert not mismatch_sql, f"金标 sql_correct 与归一化逻辑不一致: {mismatch_sql}"
