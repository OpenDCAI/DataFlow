from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.workflow.wf_pipeline_refine import create_pipeline_refine_graph

async def main() -> None:
    # ====== 1. 直接在代码内赋值参数 ======
    input_json = "/mnt/DataFlow/lz/proj/agentgroup/zewei/DFA-LG/dataflow/dataflowagent/test_pipeline.json"
    output_json = "cache_local/pipeline_refine_result.json"   # 输出路径，留空则只打印
    target = "请将Pipeline调整为只包含3个节点，简化数据流"            # 你的自然语言需求
    chat_api_url = "http://123.129.219.111:3000/v1/"
    model = "gpt-4o"
    language = "en"

    # ====== 2. 构造请求 ======
    req = DFRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=model,
        target=target,
    )
    state = DFState(request=req, messages=[])

    # ====== 3. 读取pipeline结构 ======
    if input_json and Path(input_json).exists():
        with open(input_json, "r", encoding="utf-8") as f:
            state.pipeline_structure_code = json.load(f)
    elif not state.pipeline_structure_code:
        default_path = REPO_ROOT / "dataflow" / "dataflowagent" / "test_pipeline.json"
        with open(default_path, "r", encoding="utf-8") as f:
            state.pipeline_structure_code = json.load(f)

    # ====== 4. 构建并运行workflow ======
    graph = create_pipeline_refine_graph().build()
    final_state = await graph.ainvoke(state)

    # ====== 5. 输出结果 ======
    if isinstance(final_state, dict):
        out_json = final_state.get("pipeline_structure_code", final_state)
    else:
        out_json = getattr(final_state, "pipeline_structure_code", {})

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Saved refined pipeline JSON to: {output_json}")
    else:
        print(json.dumps(out_json, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    asyncio.run(main())