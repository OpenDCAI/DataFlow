from __future__ import annotations

import asyncio, os
from typing import List
from pydantic import BaseModel, Field
from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
    local_tool_for_get_categories,
)
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
    post_process_combine_pipeline_result,
)
from dataflow.dataflowagent.agentroles.classifier import create_classifier
from dataflow.dataflowagent.agentroles.recommender import create_recommender
from dataflow.dataflowagent.agentroles.pipelinebuilder import create_pipeline_builder
from dataflow.dataflowagent.agentroles.debugger import create_code_debugger
from dataflow.dataflowagent.agentroles.rewriter import create_rewriter

from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow.dataflowagent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow import get_logger
MAX_DEBUG_ROUNDS = 3
log = get_logger()

def create_pipeline_graph() -> GenericGraphBuilder:
    # ★ 1. 图入口改成 classifier
    builder = GenericGraphBuilder(state_model=DFState, entry_point="classifier")

    # ------------------------------------------------------------------
    # Ⅰ. 工具注册
    # ------------------------------------------------------------------
    # -------- classifier 前置工具 --------
    @builder.pre_tool("sample", "classifier")
    def cls_get_sample(state: DFState):
        # 取 2 条样本
        return local_tool_for_sample(state.request, sample_size=2)["samples"]

    @builder.pre_tool("categories", "classifier")
    def cls_get_categories(state: DFState):
        return local_tool_for_get_categories()

    # -------- recommender 前置工具 --------
    @builder.pre_tool("sample", "recommender")
    def rec_get_sample(state: DFState):
        # 推荐器只拿 1 条样本
        return local_tool_for_sample(state.request, sample_size=1)["samples"]

    @builder.pre_tool("target", "recommender")
    def rec_get_target(state: DFState):
        return local_tool_for_get_purpose(state.request)

    @builder.pre_tool("operator", "recommender")
    def rec_get_operator(state: DFState):
        return get_operator_content_str(data_type=state.category.get("category", "text"))

    class GetOpInput(BaseModel):
        oplist: list = Field(description="list['xxx']的算子列表")
    @builder.post_tool("recommender")
    @tool(args_schema=GetOpInput)
    def combine_pipeline(oplist: list) -> str:
        """Combine pipeline post tool for recommender"""
        return post_process_combine_pipeline_result(oplist)

    # -------- debugger / rewriter 前置工具（保持不变） --------
    @builder.pre_tool("pipeline_code", "code_debugger")
    def get_pipeline_code_for_debug(state: DFState):
        return state.temp_data.get("pipeline_code", "")

    @builder.pre_tool("error_trace", "code_debugger")
    def get_error_trace_for_debug(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("pipeline_code", "rewriter")
    def get_pipeline_code_for_rewrite(state: DFState):
        return state.temp_data.get("pipeline_code", "")

    @builder.pre_tool("error_trace", "rewriter")
    def get_error_trace_for_rewrite(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("debug_reason", "rewriter")
    def get_debug_reason(state: DFState):
        return state.code_debug_result.get("reason", "")

    @builder.pre_tool("data_sample", "rewriter")
    def get_data_sample(state: DFState):
        return state.temp_data.get("pre_tool_results", {}).get("sample", "")

    # ------------------------------------------------------------------
    # Ⅱ. 节点实现
    # ------------------------------------------------------------------
    async def classifier_node(s: DFState) -> DFState:
        """Classifier 节点"""
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        classifier = create_classifier(tool_manager=get_tool_manager(), model_name="deepseek-v3")
        # 这里不需要 agent-mode，所以 use_agent=False
        s = await classifier.execute(s, use_agent=False)
        # 结果保存在 s.classification （BaseAgent 已写入）
        return s

    # --- recommender 子图同原来 ---
    async def build_recommender_subgraph(init_state: DFState):
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        tm = get_tool_manager()
        recommender = create_recommender(tool_manager=tm)
        init_state = await recommender.execute(init_state, use_agent=True)

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

    async def recommender_node(s: DFState) -> DFState:
        rec_graph = await build_recommender_subgraph(s)
        result = await rec_graph.ainvoke(s)
        if isinstance(result, dict):
            for k, v in result.items():         
                setattr(s, k, v)
        else:
            import dataclasses
            for f in dataclasses.fields(DFState):
                setattr(s, f.name, getattr(result, f.name))
        rec_val = s.get("recommendation", {})
        ops_list = rec_val.get("ops", []) if isinstance(rec_val, dict) else rec_val
        if not ops_list:
            raise ValueError("Recommender 没有返回有效算子列表")
        s["recommendation"] = ops_list
        return s

    async def builder_node(s: DFState) -> DFState:
        builder_agent = create_pipeline_builder()
        skip = bool(s.temp_data.get("rewritten", False))
        log.warning(f"[builder_node] skip_assemble = {skip}")
        return await builder_agent.execute(
            s,
            skip_assemble=skip,
            # file_path=s.temp_data.get("pipeline_file_path"),
            file_path= s.request.python_file_path,
            assembler_kwargs={"file_path": s.request.json_file, "chat_api_url": s.request.chat_api_url},
        )

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

    # ------------------------------------------------------------------
    # Ⅲ. 条件函数
    # ------------------------------------------------------------------
    def builder_condition(s: DFState):
        if s.request.need_debug:
            # ① 仅当调试阶段成功，才回到 builder
            if (
                s.execution_result.get("success")
                and s.temp_data.pop("debug_sample_file", None)  
            ):
                return "builder"

            # ② 正式流程成功 → 结束
            if s.execution_result.get("success"):
                return "__end__"

            # ③ 其它情况照旧
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "code_debugger"
        else:
            # 非调试模式，成功就结束，失败也结束
            return "__end__"

    # ------------------------------------------------------------------
    # Ⅳ. 组图
    # ------------------------------------------------------------------
    nodes = {
        "classifier": classifier_node,          
        "recommender": recommender_node,
        "builder": builder_node,
        "code_debugger": debugger_node,
        "rewriter": rewriter_node,
        "after_rewrite": after_rewrite_node,
    }

    edges = [
        ("classifier", "recommender"),          
        ("recommender", "builder"),
        ("code_debugger", "rewriter"),
        ("rewriter", "after_rewrite"),
        ("after_rewrite", "builder"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"builder": builder_condition})
    return builder