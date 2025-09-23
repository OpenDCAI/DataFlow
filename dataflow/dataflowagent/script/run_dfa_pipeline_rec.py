#!/usr/bin/env python
# ---- run_dataflow.py ----
from __future__ import annotations
import argparse, asyncio, os
from dataflow.dataflowagent.state import DFRequest, DFState
from pipeline_nodes import create_pipeline_graph
from IPython.display import Image, display

# ---------- 命令行 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run DataFlow Agent end-to-end")
    p.add_argument('--json-file',     required=True, help='数据集 jsonl')
    p.add_argument('--pipeline-file', required=True, help='已有 pipeline 脚本')
    p.add_argument('--chat-api-url',  default='http://123.129.219.111:3000/v1/')
    p.add_argument('--model',         default='gpt-4o')
    p.add_argument('--language',      default='en')
    p.add_argument('--target',        default='我需要 2 个reasoning的算子！')
    debug_grp = p.add_mutually_exclusive_group()
    debug_grp.add_argument('--debug',     dest='need_debug', action='store_true')
    debug_grp.add_argument('--no-debug',  dest='need_debug', action='store_false')
    p.set_defaults(need_debug=True)
    p.add_argument('--debug-rounds',  '-r', type=int, default=3)
    return p.parse_args()

# ---------- 主逻辑 ----------
async def main() -> None:
    args = parse_args()

    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        json_file=args.json_file,
        target=args.target,
        python_file_path=args.pipeline_file,
        need_debug=args.need_debug,
        max_debug_rounds=args.debug_rounds,
    )
    state = DFState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = args.need_debug

    graph = create_pipeline_graph().build()
    final_state: DFState = await graph.ainvoke(state)

    if req.need_debug:
        if final_state.get("execution_result", {}).get("success"):
            print("\n================ 最终 Pipeline 执行成功 ================\n")
            print(f"================ 可通过 python {req.python_file_path} 处理你的完整数据！ ================")
            print(final_state["execution_result"]["stdout"])
        else:
            print("\n================== 调试失败，放弃 ==================\n")
            print(final_state.get("execution_result", {}))
    else:
        print(f"================== 不需要调试，只进行组装，结果在 {req.python_file_path} ==================")

if __name__ == "__main__":
    asyncio.run(main())