from __future__ import annotations
# export PYTHONPATH=/mnt/h_h_public/lh/lz/DataFlow:$PYTHONPATH

import asyncio
import os
from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.agentroles.classifier import create_classifier
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

# file_tools 中的两个本地工具
from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
    local_tool_for_get_categories,
)
from dataflow.cli_funcs.paths import DataFlowPath
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent

async def main() -> None:
    req = DFRequest(                       
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1",
        api_key=os.getenv("DF_API_KEY", "test"),
        model="gpt-4o",
        json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    )
    state = DFState(request=req,messages=[])
    tool_manager = get_tool_manager()
    tool_manager.register_pre_tool(
        name="sample",
        func=lambda: local_tool_for_sample(req, sample_size=2)['samples'], 
        role="classifier"
    )
    tool_manager.register_pre_tool(
        name="categories",
        func=local_tool_for_get_categories,     
        role="classifier"
    )

    # 3) 创建分类器并执行 ------------------------------------------------------
    classifier = create_classifier(tool_manager=tool_manager, model_name="deepseek-v3")
    state = await classifier.execute(state, use_agent=False)
    # 4) 打印分类结果 ----------------------------------------------------------
    print("分类结果：", state.classification)
    print("state", state)


if __name__ == "__main__":
    asyncio.run(main())