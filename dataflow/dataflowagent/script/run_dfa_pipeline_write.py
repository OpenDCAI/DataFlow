#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from dataflow.dataflowagent.state import DFRequest, DFState
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


def parse_args():
    p = argparse.ArgumentParser(description="Run operator flow: match -> write -> (optional debug loop)")
    p.add_argument('--chat-api-url',  default='http://123.129.219.111:3000/v1/', help='LLM Chat API base')
    p.add_argument('--model',         default='gpt-4o', help='LLM model name')
    p.add_argument('--language',      default='en', help='Prompt output language')
    p.add_argument('--target',        required=True, help='User requirement / purpose for new operator')
    p.add_argument('--category',      default='Default', help='Operator category for matching (fallback if no classifier)')
    p.add_argument('--output',        default='', help='Optional path to write generated operator code')
    p.add_argument('--need-debug',    action='store_true', help='Enable debug loop for executing and fixing the operator')
    p.add_argument('--max-debug-rounds', type=int, default=3, help='Max debug rounds when --need-debug is set')
    return p.parse_args()


def create_operator_write_graph() -> GenericGraphBuilder:
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
                # 简化：将匹配到的算子名作为示例参考
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
        # 若用户提供了输出路径，则用作执行文件路径
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


async def main():
    args = parse_args()

    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target=args.target,
        need_debug=bool(args.need_debug),
        max_debug_rounds=int(args.max_debug_rounds),
    )
    state = DFState(request=req, messages=[])
    if args.output:
        state.temp_data["pipeline_file_path"] = args.output
    # 若用户通过参数提供了类别，也存到 temp_data 作为兜底
    if args.category:
        state.temp_data["category"] = args.category

    graph = create_operator_write_graph().build()
    final_state: DFState = await graph.ainvoke(state)

    # ---- 打印结果摘要 ----
    print("==== Match Operator Result ====")
    try:
        matched = final_state.get("matched_ops", []) if hasattr(final_state, "get") else []
        print("Matched ops:", matched)
    except Exception:
        print("Matched ops: <unavailable>")

    print("\n==== Writer Result ====")
    try:
        # 优先读取 temp_data 中被后续节点复用的代码
        code_str = final_state.temp_data.get("pipeline_code", "") if hasattr(final_state, "temp_data") else ""
        if not code_str and hasattr(final_state, "get"):
            code_str = final_state.get("draft_operator_code", "")
        # 回退：从 writer 的 agent_results 中取代码
        if not code_str and hasattr(final_state, "agent_results"):
            try:
                code_str = (
                    final_state.agent_results.get("write_the_operator", {}).get("results", {}).get("code", "")
                )
            except Exception:
                pass
        # 进一步回退：若有文件路径则读取文件内容以计算长度与预览
        if not code_str and hasattr(final_state, "temp_data"):
            from pathlib import Path
            fp = final_state.temp_data.get("pipeline_file_path")
            if fp:
                p = Path(fp)
                try:
                    if p.exists():
                        code_str = p.read_text(encoding="utf-8")
                except Exception:
                    pass
    except Exception:
        code_str = ""
    print(f"Code length: {len(code_str)}")
    if args.output:
        print(f"Saved to: {args.output}")
    else:
        # 为避免终端刷屏，仅展示前 1000 字符
        preview = (code_str or "")[:1000]
        print("Code preview:\n", preview)

    # ---- 执行结果摘要 ----
    # 汇总执行结果（鲁棒回退）：execution_result -> agent_results -> 挂载属性
    exec_res = getattr(final_state, "execution_result", {}) or {}
    if not exec_res or ("success" not in exec_res):
        if hasattr(final_state, "agent_results"):
            try:
                exec_res = final_state.agent_results.get("operator_executor", {}).get("results", {}) or exec_res
            except Exception:
                pass
    if (not exec_res or ("success" not in exec_res)) and hasattr(final_state, "operator_executor"):
        try:
            exec_res = getattr(final_state, "operator_executor", {}) or exec_res
        except Exception:
            pass
    success = bool(exec_res.get("success"))
    print("\n==== Executor Result ====")
    print("Success:", success)
    if not success:
        stderr = (exec_res.get("stderr") or exec_res.get("traceback") or "")
        print("stderr preview:\n", (stderr or "")[:500])


if __name__ == "__main__":
    asyncio.run(main())
