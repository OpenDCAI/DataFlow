from __future__ import annotations

from langgraph.graph import StateGraph
# from langgraph.prebuilt import  ToolExecutor
try:  
    from langgraph_prebuilt.tool_node import ToolNode
except ModuleNotFoundError: 
    from langgraph.prebuilt import ToolNode
from dataflow.dataflowagent.agentroles.classifier import DataContentClassifier
from dataflow.dataflowagent.state import DFState
from dataflow import get_logger

log = get_logger()

from typing import Literal

def should_continue(state) -> Literal["tools", "__end__"]:
    """
    一个路由函数，用来决定是继续调用工具还是结束流程。

    Args:
        state (dict): 当前图的状态。

    Returns:
        str: 如果需要调用工具，则返回 "tools"；否则返回 "__end__"。
    """

    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "__end__"

def build_classification_graph(state: DFState) -> StateGraph:
    """
    根据前一步 classifier.execute(..., use_agent=True) 产生的 state，
    构建 LangGraph 图。
    """
    # 从 state.temp_data 里取出实例与前置工具结果
    classifier: DataContentClassifier = state.temp_data["classifier_instance"]
    pre_tool_results = state.temp_data["pre_tool_results"]

    # 生成助手节点函数（不再捕获外层 state，而用运行时 graph_state）
    def assistant_node(graph_state: DFState):
        """
        graph_state: LangGraph 运行时传入的 DFState，
        里面会逐步累积 messages / classification 等字段。
        """
        messages = getattr(graph_state, "messages", [])

        # 第一次进入还没有 messages，需要构建
        if not messages:
            messages = classifier.build_messages(graph_state, pre_tool_results)

        # 调 LLM，可能产生 tool_calls
        return classifier.process_with_llm_for_graph(messages, graph_state)

    builder = StateGraph(DFState)

    # 1) LLM 决策节点
    builder.add_node("assistant", assistant_node)

    post_tools = classifier.get_post_tools()
    if post_tools:
        # 2) 工具执行节点
        builder.add_node("tools", ToolNode(post_tools))

        # assistant → tools  (条件跳转)
        builder.add_conditional_edges("assistant", should_continue)

        # tools → assistant  (执行完回到 LLM)
        builder.add_edge("tools", "assistant")

    builder.set_entry_point("assistant")
    return builder.compile()


from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# def build_recommendation_graph(state: DFState):
#     # 1. 取出 agent 实例和前置工具结果
#     recommender = state.temp_data["recommender_instance"]
#     pre_tool_results = state.temp_data["pre_tool_results"]
#     post_tools = recommender.get_post_tools()

#     # 2. 构建 LangGraph
#     sg = StateGraph(DFState)

#     # assistant 节点：LLM + (可选) 工具调用
#     async def assistant_node(graph_state: DFState):
#         msgs = getattr(graph_state, "messages", [])
#         if not msgs:
#             msgs = recommender.build_messages(graph_state, pre_tool_results)

#         resp = await recommender.process_with_llm_for_graph(msgs, graph_state)
#         if recommender.has_tool_calls(resp):
#             return {"messages": msgs + [resp]}        # 让工具节点去执行
#         else:
#             result = recommender.parse_result(resp.content)
#             recommender.update_state_result(graph_state, result, pre_tool_results)
#             return {"messages": msgs + [resp], "finished": True}

#     sg.add_node("assistant", assistant_node)

#     if post_tools:                         # 如果真的有后置工具
#         sg.add_node("tools", ToolNode(post_tools))
#         sg.add_conditional_edges("assistant", tools_condition)
#         sg.add_edge("tools", "assistant")

#     sg.set_entry_point("assistant")
#     return sg.compile()

# # ---------------- 执行 ----------------
# state = await recommender.execute(state, use_agent=True)  # ① 准备
# graph = build_recommendation_graph(state)                 # ② 建图
# state = await graph.ainvoke(state)                        # ③ 运行

# print("推荐结果：", state.recommendation)