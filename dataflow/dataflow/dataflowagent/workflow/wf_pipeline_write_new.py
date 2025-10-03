from __future__ import annotations
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
from dataflow.dataflowagent.state import DFState
import re
from dataflow.dataflowagent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
)
from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
)
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.agent.toolkits.operator_processor import (
    local_tool_for_get_match_operator_code,
)
from dataflow.dataflowagent.agentroles.match import create_match
from dataflow.dataflowagent.agentroles.writer import create_writer
from dataflow.dataflowagent.agentroles.debugger import create_code_debugger
from dataflow.dataflowagent.agentroles.rewriter import create_rewriter
from dataflow.dataflowagent.agentroles.append_llm_serving import create_llm_append_serving
from dataflow.dataflowagent.agentroles.instantiator import create_llm_instantiator
from dataflow.dataflowagent.agentroles.operatorexecutor import create_operator_executor


def create_operator_write_graph() -> GenericGraphBuilder:
    """Build the operator write workflow graph.

    Flow: match_operator -> write_the_operator -> (code_debugger -> rewriter)*
    """
    builder = GenericGraphBuilder(state_model=DFState, entry_point="match_operator")

    # ---------------- 前置工具：match_operator ----------------
    @builder.pre_tool("get_operator_content", "match_operator")
    def pre_get_operator_content(state: DFState):
        cat = state.category.get("category") or state.request and getattr(state.request, "category", None)
        data_type = cat or state.temp_data.get("category") or "Default"
        return get_operator_content_str(data_type=data_type)

    @builder.pre_tool("purpose", "match_operator")
    def pre_get_purpose(state: DFState):
        return local_tool_for_get_purpose(state.request)


    # ---------------- 前置工具：write_the_operator ----------------
    @builder.pre_tool("example", "write_the_operator")
    def pre_example_from_matched(state: DFState):
        """
        为写算子提供更强的 in-context 示例：
        将匹配到的所有算子源码（含 import + 类定义）拼接为示例，让 LLM 模仿项目风格。
        优先从 DFState.matched_ops 读取；若为空则回退读取 agent_results。
        """
        names: list[str] = []
        try:
            if isinstance(state.matched_ops, list) and state.matched_ops:
                names = list(dict.fromkeys(state.matched_ops))
            else:
                res = state.agent_results.get("match_operator", {}).get("results", {})
                names = list(dict.fromkeys(res.get("match_operators", []) or []))
        except Exception:
            names = []

        if not names:
            return ""

        blocks = []
        chunk = 3  # 分批聚合，避免极长提示一次性超长
        for i in range(0, len(names), chunk):
            part = names[i:i+chunk]
            try:
                blocks.append(local_tool_for_get_match_operator_code({"match_operators": part}))
            except Exception:
                continue
        code_examples = "\n\n".join([b for b in blocks if b])
        return code_examples

    @builder.pre_tool("target", "write_the_operator")
    def pre_target(state: DFState):
        return state.request.target


    # ---------------- 前置工具：code_debugger ----------------
    @builder.pre_tool("operator_code", "code_debugger")
    def dbg_get_code(state: DFState):
        return (
            state.temp_data.get("operator_code", "")
            or state.temp_data.get("pipeline_code", "")
            or getattr(state, "draft_operator_code", "")
        )

    @builder.pre_tool("FileStorage_info", "code_debugger")
    def dbg_get_filestorage_info(state: DFState):
        return """
        # -------- FileStorage --------
        self.storage = FileStorage(
            test_file_name="DataFlow/dataflow/dataflowagent/test_data.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        """

    @builder.pre_tool("llm_serving_info", "code_debugger")
    def dbg_get_llm_serving_info(state: DFState):
        return """
        # -------- LLM Serving (Remote) --------
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key="DF_API_KEY",
            model_name="gpt-4o",
            max_workers=100,
        )
        """

    @builder.pre_tool("data_sample", "code_debugger")
    def dbg_get_data_sample(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            samples = stats.get("samples", []) if isinstance(stats, dict) else []
            import json
            return json.dumps(samples, ensure_ascii=False, indent=2) if samples else "[]"
        except Exception:
            return "[]"
    
    @builder.pre_tool("target", "code_debugger")
    def dbg_get_target(state: DFState):
        return state.request.target


    # ---------------- 前置工具：rewriter ----------------
    @builder.pre_tool("operator_code", "rewriter")
    def rw_get_code(state: DFState):
        return state.temp_data.get("operator_code", "")

    @builder.pre_tool("instance_code", "rewriter")
    def rw_get_instance_code(state: DFState):
        return state.temp_data.get("instance_code", "")
    
    @builder.pre_tool("error_trace", "rewriter")
    def rw_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("debug_reason", "rewriter")
    def rw_get_reason(state: DFState):
        return state.code_debug_result.get("reason", "")

    @builder.pre_tool("data_sample", "rewriter")
    def rw_get_data_sample(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("target", "rewriter")
    def rw_get_target(state: DFState):
        return getattr(state.request, "target", "")


    # ---------------- 节点实现 ----------------
    async def match_node(s: DFState) -> DFState:
        agent = create_match()
        return await agent.execute(s, use_agent=False)

    async def write_node(s: DFState) -> DFState:
        agent = create_writer()
        return await agent.execute(s, use_agent=False)

    async def operator_executor(s: DFState) -> DFState:
        """
        执行当前生成/重写后的算子代码：
        - 将代码写入文件并运行
        - 成功判定：能正确处理 jsonl 测试数据并写入结果文件
        """
        agent = create_operator_executor(tool_manager=get_tool_manager())
        s2 = await agent.execute(s)
        # 判定：缓存结果文件存在且非空（优先尝试读取为 JSONL）
        from pathlib import Path
        p = Path("./cache_local/dataflow_cache_step_step1.jsonl")
        success = False
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_json(str(p), lines=True)
                success = (not df.empty)
            except Exception:
                success = (p.stat().st_size > 0)
        s2.execution_result = s2.execution_result or {}
        s2.execution_result["success"] = bool(success)
        s2.execution_result["file_path"] = s2.temp_data.get("pipeline_file_path", s2.execution_result.get("file_path", ""))
        return s2

    async def llm_instantiate(s: DFState) -> DFState:
        """
        调试（生成实例化代码）：
        - 利用 LLM（InstantiateAgent）基于已写好的算子类，生成实例化与运行入口代码；
        - 将 FileStorage、LLM Serving、sample data、target 等上下文通过 pre_tools 注入；
        - 输出写入 state.temp_data['pipeline_code']，供 operator_executor 执行。
        """
        tm = get_tool_manager()

        # 基础上下文
        operator_code = (
            s.temp_data.get("operator_code", "")
            or s.temp_data.get("pipeline_code", "")
            or getattr(s, "draft_operator_code", "")
        )
        fs_info = dbg_get_filestorage_info(s)
        llm_info = dbg_get_llm_serving_info(s)

        # 将 FileStorage/LLM Serving 作为注释拼接进 pipeline_code，帮助 LLM 生成 runner
        context_hint = (
            "\n\n# ==== Context: FileStorage Suggestion ====\n" + str(fs_info or "") +
            "\n\n# ==== Context: LLM Serving Suggestion ====\n" + str(llm_info or "") + "\n"
        )
        pipeline_code_seed = (operator_code or "") + context_hint

        # 测试数据与可选输入键
        from types import SimpleNamespace as _SN
        default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
        test_data_path = getattr(s.request, "json_file", "") or default_test_file
        stats = local_tool_for_sample(_SN(json_file=test_data_path), sample_size=2)
        example_data = stats.get("samples", []) if isinstance(stats, dict) else []
        available_keys = stats.get("available_keys", []) if isinstance(stats, dict) else []
        preselected_key = available_keys[0] if available_keys else ""
        target = getattr(s.request, "target", "")

        # 将上述作为 llm_instantiate 角色的前置工具注册，然后调用 InstantiateAgent
        tm.register_pre_tool(name="pipeline_code", role="llm_instantiate", func=lambda c=pipeline_code_seed: c)
        tm.register_pre_tool(name="target", role="llm_instantiate", func=lambda t=target: t)
        tm.register_pre_tool(name="example_data", role="llm_instantiate", func=lambda d=example_data: d)
        tm.register_pre_tool(name="available_keys", role="llm_instantiate", func=lambda ks=available_keys: ks)
        tm.register_pre_tool(name="preselected_input_key", role="llm_instantiate", func=lambda k=preselected_key: k)
        tm.register_pre_tool(name="test_data_path", role="llm_instantiate", func=lambda p=test_data_path: p)
        # 传入明确的初始化片段，强约束使用相同参数
        tm.register_pre_tool(name="llm_serving_info", role="llm_instantiate", func=lambda s=llm_info: s)
        tm.register_pre_tool(name="FileStorage_info", role="llm_instantiate", func=lambda f=fs_info: f)

        agent = create_llm_instantiator(tool_manager=tm)
        return await agent.execute(s, use_agent=True)

    async def rewriter_node(s: DFState) -> DFState:
        agent = create_rewriter(tool_manager=get_tool_manager(), model_name="o3")
        s2 = await agent.execute(s, use_agent=True)
        # 调整轮次计数，便于条件边生效
        try:
            s2.temp_data["round"] = s2.temp_data.get("round", 0) + 1
        except Exception:
            pass
        return s2



    # ---------------- 条件边（复用 pipeline 的循环思路） ----------------
    def exec_condition(s: DFState):
        if s.request.need_debug:
            if s.execution_result.get("success"):
                return "__end__"
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "rewriter"
        else:
            return "__end__"

    nodes = {
        "match_operator": match_node,
        "write_the_operator": write_node,
        "operator_executor": operator_executor,
        "llm_instantiate": llm_instantiate,
        "rewriter": rewriter_node,
    }
    edges = [
        ("match_operator", "write_the_operator"),
        ("write_the_operator", "llm_instantiate"),
        ("llm_instantiate", "operator_executor"),
        ("rewriter", "llm_instantiate"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"operator_executor": exec_condition})
    return builder
