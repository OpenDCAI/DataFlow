#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import os
import re
import io
import contextlib
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
    p.add_argument('--json-file',     default='', help='Path to test jsonl file used in debug run')
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

    from dataflow.cli_funcs.paths import DataFlowPath

    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target=args.target,
        need_debug=bool(args.need_debug),
        max_debug_rounds=int(args.max_debug_rounds),
        # 默认使用 dataflowagent 下的 10 条测试数据
        json_file=(args.json_file or f"{DataFlowPath.get_dataflow_dir()}/dataflowagent/test_data.jsonl"),
    )
    state = DFState(request=req, messages=[])
    if args.output:
        state.temp_data["pipeline_file_path"] = args.output
    # 若用户通过参数提供了类别，也存到 temp_data 作为兜底
    if args.category:
        state.temp_data["category"] = args.category

    # 显式初始化调试轮次
    state.temp_data["round"] = 0

    graph = create_operator_write_graph().build()
    # LangGraph 默认 recursion_limit=25，当 need_debug 且 max_debug_rounds 较大时容易超限。
    # 计算一个保守上限：主链 4 步 + 每轮 5 步 * 轮次 + buffer 5。
    recursion_limit = 4 + 5 * int(args.max_debug_rounds) + 5
    final_state: DFState = await graph.ainvoke(state, config={"recursion_limit": recursion_limit})

    # ---- 打印结果摘要 ----
    print("==== Match Operator Result ====")
    try:
        if isinstance(final_state, dict):
            matched = final_state.get("matched_ops")
            if not matched:
                matched = (
                    final_state.get("agent_results", {})
                    .get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        else:
            matched = getattr(final_state, "matched_ops", [])
            if not matched and hasattr(final_state, "agent_results"):
                matched = (
                    final_state.agent_results.get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        print("Matched ops:", matched or [])
    except Exception:
        print("Matched ops: <unavailable>")

    print("\n==== Writer Result ====")
    try:
        # 兼容 dict 与 DFState 两种返回形态
        if isinstance(final_state, dict):
            code_str = (
                final_state.get("temp_data", {}).get("pipeline_code", "")
                or final_state.get("draft_operator_code", "")
                or final_state.get("agent_results", {}).get("write_the_operator", {}).get("results", {}).get("code", "")
            )
            if not code_str:
                fp = final_state.get("temp_data", {}).get("pipeline_file_path")
                if fp:
                    from pathlib import Path
                    p = Path(fp)
                    try:
                        if p.exists():
                            code_str = p.read_text(encoding="utf-8")
                    except Exception:
                        pass
        else:
            code_str = (
                getattr(final_state, "temp_data", {}).get("pipeline_code", "")
                or getattr(final_state, "draft_operator_code", "")
                or getattr(getattr(final_state, "agent_results", {}), "get", lambda *_: {})("write_the_operator", {}).get("results", {}).get("code", "")
            )
            if not code_str:
                fp = getattr(final_state, "temp_data", {}).get("pipeline_file_path")
                if fp:
                    from pathlib import Path
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
        # 为避免终端刷屏，仅展示前 3000 字符
        preview = (code_str or "")[:3000]
        print("Code preview:\n", preview)

    # ---- Debug runtime 移至 workflow 的 instantiate_operator_main_node ----
    # 入口脚本不再负责内联实例化执行，避免过度复杂。

    # ---- 执行结果摘要 ----
    # 汇总执行结果（鲁棒回退）：execution_result -> agent_results -> 挂载属性
    # 兼容 dict 与 DFState 两种返回形态
    if isinstance(final_state, dict):
        exec_res = final_state.get("execution_result", {}) or {}
        if not exec_res or ("success" not in exec_res):
            exec_res = final_state.get("agent_results", {}).get("operator_executor", {}).get("results", {}) or exec_res
    else:
        exec_res = getattr(final_state, "execution_result", {}) or {}
        if (not exec_res or ("success" not in exec_res)) and hasattr(final_state, "agent_results"):
            exec_res = final_state.agent_results.get("operator_executor", {}).get("results", {}) or exec_res
    success = bool(exec_res.get("success"))
    print("\n==== Execution Result (instantiate) ====")
    print("Success:", success)
    if not success:
        stderr = (exec_res.get("stderr") or exec_res.get("traceback") or "")
        print("stderr preview:\n", (stderr or "")[:500])

    # ---- 调试实例化输出预览（来自 instantiate_operator_main_node） ----
    try:
        dbg = None
        if isinstance(final_state, dict):
            dbg = (final_state.get("temp_data") or {}).get("debug_runtime")
        else:
            dbg = getattr(final_state, "temp_data", {}).get("debug_runtime")
        if dbg:
            print("\n==== Debug Runtime Preview ==== ")
            ik = dbg.get("input_key")
            ak = dbg.get("available_keys")
            print("input_key:", ik)
            if ak:
                print("available_keys:", ak)
            stdout_pv = (dbg.get("stdout") or "")[:1000]
            stderr_pv = (dbg.get("stderr") or "")[:1000]
            if stdout_pv:
                print("[debug stdout]\n", stdout_pv)
            if stderr_pv:
                print("[debug stderr]\n", stderr_pv)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
