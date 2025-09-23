from __future__ import annotations

import asyncio
import os
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.dataflowagent.state import DFRequest, DFState
from pipeline_nodes import create_pipeline_graph
from IPython.display import Image, display

async def main() -> None:
    # 初始请求
    DATAFLOW_DIR = DataFlowPath.get_dataflow_dir().parent
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model="gpt-4o",
        json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        target="我需要 2 个reasoning的算子！",
        python_file_path = f"{DATAFLOW_DIR}/dataflow/dataflowagent/tests/my_pipeline.py",  # pipeline的输出脚本路径
        need_debug = True, #是否需要Debug
        max_debug_rounds = 3, #Debug的轮次数量
    )
    state = DFState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = True

    # 构建并运行图
    graph_builder = create_pipeline_graph()
    graph = graph_builder.build()
    try:
        png_image = graph.get_graph().draw_mermaid_png()
        display(Image(png_image))

        with open("my_graph.png", "wb") as f:
            f.write(png_image)
        print("\n图已保存为 my_graph.png")

    except Exception as e:
        print(f"生成PNG失败，请确保已正确安装 pygraphviz 和 Graphviz：{e}")
    
    final_state: DFState = await graph.ainvoke(state)

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

    


if __name__ == "__main__":
    asyncio.run(main())