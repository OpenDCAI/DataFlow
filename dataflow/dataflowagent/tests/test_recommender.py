from __future__ import annotations

import asyncio
from dataclasses import Field
import os
from typing import List, Literal

from langchain_core.tools import Tool
from pydantic import create_model
from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.agentroles.recommender import create_recommender
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
from pydantic import BaseModel, Field
from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample
)
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    get_operator_content,
    local_tool_for_get_purpose,
    post_process_combine_pipeline_result,
    get_operator_content_str
)
from langgraph.graph import StateGraph
try:  
    from langgraph_prebuilt.tool_node import ToolNode,tools_condition
except ModuleNotFoundError: 
    from langgraph.prebuilt import ToolNode,tools_condition
from dataflow.cli_funcs.paths import DataFlowPath
from langchain.tools import tool
from dataflow import get_logger

log = get_logger()
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent

# def should_continue(state) -> Literal["tools", "__end__"]:
#     """
#     一个路由函数，用来决定是继续调用工具还是结束流程。

#     Args:
#         state (dict): 当前图的状态。

#     Returns:
#         str: 如果需要调用工具，则返回 "tools"；否则返回 "__end__"。
#     """
#     last_message = state.messages[-1]

#     if last_message.tool_calls:
#         return "tools"
#     else:
#         return "__end__"

async def main() -> None:
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("DF_API_KEY", " "),
        model="gpt-4o",
        json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        target = "我需要3个算子，其中不需要去重的算子！"
    )
    state = DFState(request=req,messages=[])

    tm = get_tool_manager()

    # ---------- 前置工具：sample / target / operator -------------------------
    tm.register_pre_tool(
        name="sample",
        func=lambda: local_tool_for_sample(req, sample_size=3)["samples"],
        role="recommender",
    )
    tm.register_pre_tool(
        name="target",
        func=lambda: local_tool_for_get_purpose(req),
        role="recommender",
    )
    tm.register_pre_tool(
        name="operator",
        # func= get_operator_content,
        # func = lambda: get_operator_content(data_type= state.classification.get('category','Default')),
        func = lambda: get_operator_content_str(data_type= "text2sql"),
        role="recommender",
    )
    class GetOpInput(BaseModel):
        """定义 GetOpInput 工具的输入参数结构"""
        oplist: list = Field(description="传入一个推荐的算子列表，自动返回算子的管线；")
    @tool(args_schema=GetOpInput)
    def test_function_call(oplist: list) -> str:
        return post_process_combine_pipeline_result(oplist)
    post_tool = test_function_call
    tm.register_post_tool(post_tool, role="recommender")


    # 3) 创建 Recommender 并执行 ----------------------------------------------
    # 若希望让 LLM 有机会调用后置工具，需要 use_agent=True
    recommender = create_recommender(tool_manager=tm)
    state = await recommender.execute(state, use_agent=True)


    def build_recommendation_graph(state: DFState):
        # 1. 取出 agent 实例和前置工具结果
        recommender = state.temp_data["recommender_instance"]
        pre_tool_results = state.temp_data["pre_tool_results"]
        post_tools = recommender.get_post_tools()

        # 2. 构建 LangGraph
        sg = StateGraph(DFState)
        assistant_node = recommender.create_assistant_node_func(state, pre_tool_results)
        sg.add_node("assistant", assistant_node)

        if post_tools:                  
            sg.add_node("tools", ToolNode(post_tools))
            sg.add_conditional_edges("assistant", tools_condition)
            sg.add_edge("tools", "assistant")

        sg.set_entry_point("assistant")
        return sg.compile()

    # ---------------- 执行 ----------------
    graph = build_recommendation_graph(state)                 # ② 建图
    result_dict = await graph.ainvoke(state)  # 返回字典
    
    print("推荐结果：", result_dict.get('recommendation', {}))
    print("推荐结果：", state.recommendation)
    print('state:',state)

    # 4) 输出结果 -------------------------------------------------------------
    # print("推荐结果：", state.recommendation)

if __name__ == "__main__":
    asyncio.run(main())
