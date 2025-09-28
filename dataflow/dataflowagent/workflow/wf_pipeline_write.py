from __future__ import annotations

from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
)
from dataflow.dataflowagent.agentroles.match import create_match
from dataflow.dataflowagent.agentroles.writer import create_writer
from dataflow.dataflowagent.agentroles.operatorexecutor import create_operator_executor
from dataflow.dataflowagent.agentroles.debugger import create_code_debugger
from dataflow.dataflowagent.agentroles.rewriter import create_rewriter


def create_operator_write_graph() -> GenericGraphBuilder:
    """Build the operator write workflow graph.

    Flow: match_operator -> write_the_operator -> operator_executor
          -> (code_debugger -> rewriter -> after_rewrite -> operator_executor)*
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
        try:
            if isinstance(state.matched_ops, list) and state.matched_ops:
                return "; ".join(map(str, state.matched_ops))
        except Exception:
            pass
        return ""

    @builder.pre_tool("target", "write_the_operator")
    def pre_target(state: DFState):
        return state.request.target

    # ---------------- 调试相关前置工具（对齐 pipeline 复用） ----------------
    @builder.pre_tool("pipeline_code", "code_debugger")
    def dbg_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("error_trace", "code_debugger")
    def dbg_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("pipeline_code", "rewriter")
    def rw_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("error_trace", "rewriter")
    def rw_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("debug_reason", "rewriter")
    def rw_get_reason(state: DFState):
        return state.code_debug_result.get("reason", "")

    # ---------------- 节点实现 ----------------
    async def match_node(s: DFState) -> DFState:
        agent = create_match()
        return await agent.execute(s, use_agent=False)

    async def write_node(s: DFState) -> DFState:
        agent = create_writer()
        return await agent.execute(s, use_agent=False)

    async def executor_node(s: DFState) -> DFState:
        agent = create_operator_executor()
        return await agent.execute(s, file_path=s.temp_data.get("pipeline_file_path"))

    async def debugger_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        debugger = create_code_debugger(tool_manager=get_tool_manager())
        return await debugger.execute(s, use_agent=True)

    async def rewriter_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name="o3")
        return await rewriter.execute(s, use_agent=True)

    def after_rewrite_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name="o3")
        return rewriter.after_rewrite(s)

    # ---------------- 条件边（复用 pipeline 的循环思路） ----------------
    def exec_condition(s: DFState):
        if s.request.need_debug:
            if s.execution_result.get("success"):
                return "__end__"
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "code_debugger"
        else:
            return "__end__"

    nodes = {
        "match_operator": match_node,
        "write_the_operator": write_node,
        "operator_executor": executor_node,
        "code_debugger": debugger_node,
        "rewriter": rewriter_node,
        "after_rewrite": after_rewrite_node,
    }
    edges = [
        ("match_operator", "write_the_operator"),
        ("write_the_operator", "operator_executor"),
        ("code_debugger", "rewriter"),
        ("rewriter", "after_rewrite"),
        ("after_rewrite", "operator_executor"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"operator_executor": exec_condition})
    return builder

