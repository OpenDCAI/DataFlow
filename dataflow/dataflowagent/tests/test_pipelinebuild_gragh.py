# dataflow/dataflowagent/tests/test_pipelinebuilder.py
# -----------------------------------------------------
# 全流程：Recommender → (Builder ⇆ Debugger ⇆ Rewriter)n
# -----------------------------------------------------
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from langchain.tools import tool

try:
    from langgraph_prebuilt.tool_node import ToolNode, tools_condition
except ModuleNotFoundError:
    from langgraph.prebuilt import ToolNode, tools_condition

# ------------------------- DataFlow 相关 ----------------------------
from dataflow import get_logger
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
from dataflow.dataflowagent.toolkits.basetool.file_tools import local_tool_for_sample
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
    post_process_combine_pipeline_result,
)
from dataflow.dataflowagent.agentroles.recommender import create_recommender
from dataflow.dataflowagent.agentroles.pipelinebuilder import create_pipeline_builder
from dataflow.dataflowagent.agentroles.debugger import create_code_debugger
from dataflow.dataflowagent.agentroles.rewriter import create_rewriter

log = get_logger()

# ----------------------------- 常量 ----------------------------------
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent

MAX_DEBUG_ROUNDS = 3      # 最多执行→调试→重写 的轮次
DEBUG_SAMPLE_LINES = 10   # PipelineBuilder 调试模式截取的行数


# --------------------------- 辅助工具 --------------------------------
def _safe_register_pre_tool(tm, *, name: str, role: str, func):
    """
    兼容不同版本 ToolManager：有 override 参数用 override，没有就直接覆盖
    """
    kwargs = dict(name=name, role=role, func=func)
    try:
        tm.register_pre_tool(**kwargs, override=True)
    except TypeError:
        tm.register_pre_tool(**kwargs)


def register_debug_tools(tm, state: DFState):
    """
    每一轮重新覆盖 Debugger / Rewriter 所需的前置工具，
    保证读到最新的代码字符串 & 错误堆栈。
    """
    # ---------- CodeDebugger ----------
    _safe_register_pre_tool(
        tm,
        name="pipeline_code",
        role="code_debugger",
        func=lambda: state.temp_data.get("pipeline_code", ""),
    )
    _safe_register_pre_tool(
        tm,
        name="error_trace",
        role="code_debugger",
        func=lambda: state.execution_result.get("stderr", "")
        or state.execution_result.get("traceback", ""),
    )

    # ---------- Rewriter --------------
    _safe_register_pre_tool(
        tm,
        name="pipeline_code",
        role="rewriter",
        func=lambda: state.temp_data.get("pipeline_code", ""),
    )
    _safe_register_pre_tool(
        tm,
        name="error_trace",
        role="rewriter",
        func=lambda: state.execution_result.get("stderr", "")
        or state.execution_result.get("traceback", ""),
    )
    _safe_register_pre_tool(
        tm,
        name="debug_reason",
        role="rewriter",
        func=lambda: state.code_debug_result.get("reason", ""),
    )
    _safe_register_pre_tool(
        tm,
        name="data_sample",
        role="rewriter",
        func=lambda: state.temp_data.get("pre_tool_results", {}).get("sample", ""),
    )


# -------------------------------------------------------------------- #
#                                主流程                                 #
# -------------------------------------------------------------------- #
async def main() -> None:
    # --------------------- 0. 请求 & State ----------------------------
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model="gpt-4o",
        json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        target="我需要 3 个算子，其中不需要去重的算子！",
    )
    state = DFState(request=req, messages=[])

    tm = get_tool_manager()

    # --------------------- 1. Recommender -----------------------------
    # 1.1 注册前置 / 后置工具
    tm.register_pre_tool(
        name="sample",
        role="recommender",
        func=lambda: local_tool_for_sample(req, sample_size=1)["samples"],
    )
    tm.register_pre_tool(
        name="target",
        role="recommender",
        func=lambda: local_tool_for_get_purpose(req),
    )
    tm.register_pre_tool(
        name="operator",
        role="recommender",
        func=lambda: get_operator_content_str(data_type="text2sql"),
    )

    class GetOpInput(BaseModel):
        oplist: list = Field(description="传入一个推荐算子列表，自动返回算子的管线；")

    @tool(args_schema=GetOpInput)
    def combine_pipeline(oplist: list) -> str:
        """
        Combine a list of recommended operators into a full pipeline string.

        Args:
            oplist (list): Recommended operator names.

        Returns:
            str: The assembled pipeline string.
        """
        return post_process_combine_pipeline_result(oplist)

    tm.register_post_tool(combine_pipeline, role="recommender")

    recommender = create_recommender(tool_manager=tm)
    state = await recommender.execute(state, use_agent=True)

    # 1.2 让 LLM 在 LangGraph 中调用后置工具
    def build_rec_graph(init_state: DFState):
        rec_inst = init_state.temp_data["recommender_instance"]
        pre_results = init_state.temp_data["pre_tool_results"]
        post_tools = rec_inst.get_post_tools()

        sg = StateGraph(DFState)
        sg.add_node("assistant", rec_inst.create_assistant_node_func(init_state, pre_results))
        if post_tools:
            sg.add_node("tools", ToolNode(post_tools))
            sg.add_conditional_edges("assistant", tools_condition)
            sg.add_edge("tools", "assistant")
        sg.set_entry_point("assistant")
        return sg.compile()

    await build_rec_graph(state).ainvoke(state)

    # 1.3 解析推荐算子
    rec_val = state.recommendation
    ops_list: List[str] = rec_val.get("ops", []) if isinstance(rec_val, dict) else rec_val
    if not ops_list:
        raise ValueError("Recommender 没有返回有效算子列表")
    state.recommendation = ops_list
    print(f"\n>>> Recommender 推荐算子列表: {ops_list}\n")

    # --------------------- 2. 三大 Agent 实例 -------------------------
    builder = create_pipeline_builder()
    debugger = create_code_debugger(tool_manager=tm)
    rewriter = create_rewriter(tool_manager=tm, model_name="o3")

    # --------------------- 3. 构造 LangGraph -------------------------
    sg = StateGraph(DFState)
    
    # ---- 3.1 Builder 节点 ------------------------------------------
    async def builder_node(s: DFState) -> DFState:
        s = await builder.execute(
            s,
            skip_assemble=bool(s.temp_data.get("rewritten", False)),
            file_path=s.temp_data.get("pipeline_file_path"),
            assembler_kwargs={
                "file_path": s.request.json_file,
                "chat_api_url": s.request.chat_api_url,
            },
        )
        # 每轮执行完，刷新调试工具
        register_debug_tools(tm, s)
        return s

    sg.add_node("builder", builder_node)
    

    # ---- 3.2 判断 Builder 结果 -------------------------------------
    def builder_condition(s: DFState):
        if s.execution_result.get("success"):
            return "__end__"           # 运行成功 → 终止
        if s.temp_data.get("round", 0) >= MAX_DEBUG_ROUNDS:
            return "__end__"           # 超出调试次数 → 终止
        return "debugger"         # 失败 → 调试

    sg.add_conditional_edges("builder", builder_condition)

    async def debugger_node(s: DFState) -> DFState:
        return await debugger.execute(s, use_agent=True)

    async def rewriter_node(s: DFState) -> DFState:
        return await rewriter.execute(s, use_agent=True)
    
    
    sg.add_node("debugger", debugger_node)
    sg.add_node("rewriter", rewriter_node)
    # sg.add_edge("builder", "debugger")
    sg.add_edge("debugger", "rewriter")
    sg.add_node("after_rewrite", rewriter.after_rewrite)
    sg.add_edge("rewriter", "after_rewrite")
    sg.add_edge("after_rewrite", "builder")
    sg.set_entry_point("builder")
    pipeline_graph = sg.compile()

    final_state = await pipeline_graph.ainvoke(state)

    if req.need_debug:
        if final_state.get("execution_result", {}).get("success"):
            print("\n================ 最终 Pipeline 执行成功 ================\n")
            print(f"================ 可通过 python {req.python_file_path} 处理你的完整数据！ ================")
            print(final_state["execution_result"]["stdout"])
        else:
            print("\n================== 调试失败，放弃 ==================\n")
            print(final_state.get("execution_result", {}))
            assert final_state.get("execution_result", {}).get("success") is True
            assert isinstance(final_state.get("code_debug_result", {}), dict)
            assert isinstance(final_state.get("rewrite_result", {}), dict)
    else:
        print(f"================== 不需要调试，只进行组装，结果在 {req.python_file_path} ==================")

# --------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())