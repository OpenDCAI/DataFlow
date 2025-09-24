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


def parse_args():
    p = argparse.ArgumentParser(description="Run two-node graph: match_operator -> write_the_operator")
    p.add_argument('--chat-api-url',  default='http://123.129.219.111:3000/v1/', help='LLM Chat API base')
    p.add_argument('--model',         default='gpt-4o', help='LLM model name')
    p.add_argument('--language',      default='en', help='Prompt output language')
    p.add_argument('--target',        required=True, help='User requirement / purpose for new operator')
    p.add_argument('--category',      default='Default', help='Operator category for matching (fallback if no classifier)')
    p.add_argument('--output',        default='', help='Optional path to write generated operator code')
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

    # ---------------- 节点实现 ----------------
    async def match_node(s: DFState) -> DFState:
        agent = create_match()
        return await agent.execute(s, use_agent=False)

    async def write_node(s: DFState) -> DFState:
        agent = create_writer()
        return await agent.execute(s, use_agent=False)

    nodes = {
        "match_operator": match_node,
        "write_the_operator": write_node,
    }
    edges = [("match_operator", "write_the_operator")]

    builder.add_nodes(nodes).add_edges(edges)
    return builder


async def main():
    args = parse_args()

    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target=args.target,
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
        code_str = final_state.get("draft_operator_code", "") if hasattr(final_state, "get") else ""
    except Exception:
        code_str = ""
    print(f"Code length: {len(code_str)}")
    if args.output:
        print(f"Saved to: {args.output}")
    else:
        # 为避免终端刷屏，仅展示前 1000 字符
        preview = (code_str or "")[:1000]
        print("Code preview:\n", preview)


if __name__ == "__main__":
    asyncio.run(main())
