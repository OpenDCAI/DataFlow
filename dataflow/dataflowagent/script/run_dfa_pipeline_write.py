#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.workflow.wf_pipeline_write import create_operator_write_graph


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


"""
Entry script for operator write workflow.
Only keeps CLI parsing, DFState creation, and running the workflow graph.
Graph construction and node/tool definitions live in workflow/wf_pipeline_write.py
"""


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
