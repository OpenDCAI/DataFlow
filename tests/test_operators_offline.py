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

from dataflow_finance_text2sql import (  # noqa: E402
    FinanceSchemaLinker,
    FinanceSQLGenerator,
    FinanceSQLRepairRefiner,
    FinanceSQLSyntaxFilter,
)
from dataflow.utils.registry import OPERATOR_REGISTRY  # noqa: E402


EXPECTED_OPS = [
    "FinanceSchemaLinker",
    "FinanceSQLGenerator",
    "FinanceSQLSyntaxFilter",
    "FinanceSQLRepairRefiner",
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
    """按 system_prompt 关键词区分三类调用：Schema 关联 / SQL 生成 / SQL 修复。"""

    def __init__(self, bad_sql=False):
        self.bad_sql = bad_sql
        self.calls = 0

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls += len(user_inputs)
        out = []
        for _ in user_inputs:
            if "Schema 关联专家" in system_prompt:  # linker
                out.append('{"tables": ["revenue"], "columns": ["revenue.company"], "join_hints": [], "notes": "stub"}')
            elif "SQL 修复专家" in system_prompt:  # repairer
                out.append("```sql\nSELECT company FROM revenue WHERE year = 2024\n```")
            else:  # SQL 生成
                if self.bad_sql:
                    out.append("```sql\nSELEC company FRM revenue\n```")
                else:
                    out.append("```sql\nSELECT company, revenue FROM revenue WHERE year = 2024\n```")
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
        {"question": "查询2024年营收最高的公司", "schema": "表 revenue(company TEXT, year INT, revenue NUMERIC)"},
        {"question": "", "schema": "表 revenue(company TEXT, year INT, revenue NUMERIC)"},
    ]
)


def test_linker_generate_e2e():
    llm = FakeLLM()
    storage = FakeStorage(BASE_DF.copy())

    FinanceSchemaLinker(llm_serving=llm).run(
        storage, input_question="question", input_schema="schema", output_schema_context="schema_context"
    )
    assert storage.df["schema_context"].iloc[0].startswith("{")

    FinanceSQLGenerator(llm_serving=llm).run(
        storage, input_question="question", input_schema_context="schema_context", output_sql="sql"
    )
    assert storage.df["sql"].iloc[0].upper().startswith("SELECT")


def test_syntax_filter_keeps_valid_drops_invalid():
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"sql": "SELECT company FROM revenue WHERE year = 2024"},
                {"sql": "SELECT ( FROM revenue"},
                {"sql": ""},
                {"sql": "DROP TABLE revenue"},
            ]
        )
    )
    FinanceSQLSyntaxFilter().run(storage, input_sql="sql", output_sql_error="sql_error")
    assert len(storage.df) == 1  # 仅保留合法 SELECT
    assert storage.df["sql_error"].iloc[0] == ""


def test_repair_refiner_fixes_error_rows():
    llm = FakeLLM()
    storage = FakeStorage(
        pd.DataFrame(
            [
                {"sql": "SELEC x", "sql_error": "syntax error", "schema_context": "{}"},
                {"sql": "SELECT 1", "sql_error": "", "schema_context": "{}"},
            ]
        )
    )
    FinanceSQLRepairRefiner(llm_serving=llm).run(
        storage,
        input_sql="sql",
        input_sql_error="sql_error",
        input_schema_context="schema_context",
        output_sql_repaired="sql_repaired",
    )
    assert storage.df["sql_repaired"].iloc[0].upper().startswith("SELECT")
    assert storage.df["sql_repaired"].iloc[1] == "SELECT 1"  # 无报错原样保留


def test_evalset_seed_format():
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "finance_text2sql_eval_set.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 30
    required = {"id", "difficulty", "question", "schema", "gold_sql"}
    for row in rows:
        assert required.issubset(row.keys()), f"样本 {row.get('id')} 缺字段"
        assert row["difficulty"] in {"easy", "medium", "hard"}
        assert row["gold_sql"].strip().upper().startswith(("SELECT", "WITH")), (
            f"样本 {row['id']} gold_sql 必须是只读查询"
        )
