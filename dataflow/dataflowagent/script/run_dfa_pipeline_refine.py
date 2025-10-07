#!/usr/bin/env python
from __future__ import annotations
import argparse, asyncio, json, os, sys
from pathlib import Path

# 将仓库根目录加入 sys.path，便于以包方式导入
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.workflow.wf_pipeline_refine import create_pipeline_refine_graph


def parse_args():
    p = argparse.ArgumentParser(description="Run JSON pipeline refine workflow")
    p.add_argument('--input_json', default='/mnt/DataFlow/lz/proj/agentgroup/zewei/DFA-LG/dataflow/dataflowagent/test_pipeline.json', help='初始 pipeline JSON 文件路径；留空则从 DFState.pipeline_structure_code 读取')
    p.add_argument('--output_json', required=False, default='cache_local/pipeline_refine_result.json', help='保存更新后的 pipeline JSON 路径')
    p.add_argument('--target', required=True, help='自然语言需求')
    p.add_argument('--chat-api-url', default='http://123.129.219.111:3000/v1/')
    p.add_argument('--model', default='gpt-4o')
    p.add_argument('--language', default='en')
    return p.parse_args()


async def main():
    args = parse_args()

    # 构造请求
    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target=args.target,
    )
    state = DFState(request=req, messages=[])

    # 读取 pipeline JSON：如果命令行提供 input_json 则以文件为准；
    # 否则默认从 state.pipeline_structure_code 读取（允许外部代码预先注入），
    # 若 state 中为空，则回落到仓库自带的示例 JSON。
    if args.input_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            state.pipeline_structure_code = json.load(f)
    elif not state.pipeline_structure_code:
        default_path = Path(__file__).resolve().parents[2] / 'dataflow' / 'dataflowagent' / 'test_pipeline.json'
        with open(default_path, 'r', encoding='utf-8') as f:
            state.pipeline_structure_code = json.load(f)

    graph = create_pipeline_refine_graph().build()
    final_state = await graph.ainvoke(state)

    # 输出结果：若提供输出路径则落盘，否则打印到控制台
    if isinstance(final_state, dict):
        out_json = final_state.get("pipeline_structure_code", final_state)
    else:
        out_json = getattr(final_state, "pipeline_structure_code", {})
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Saved refined pipeline JSON to: {args.output_json}")
    else:
        print(json.dumps(out_json, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
