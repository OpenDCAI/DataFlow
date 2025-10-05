from __future__ import annotations
import argparse, asyncio, os
from langgraph.graph import StateGraph, START, END
import sys
sys.path.append('/mnt/DataFlow/zks/DataFlow')
from dataflow.dataflowagent.state import DataCollectionRequest, DataCollectionState
from dataflow.dataflowagent.agentroles.datacollector import DataCollector, data_collection
from dataflow.dataflowagent.agentroles.dataconvertor import DataConvertor, data_conversion

async def main() -> None:
    req = DataCollectionRequest(
        api_key="sk-J4OU0nswdAQEmN7y7pS9ytPedSvEC8NXCOhuBX5GIz3dXz3c",
        target="我需要一些金融和法律数据"
    )


    state = DataCollectionState(request=req)

    graph_builder = StateGraph(DataCollectionState)
    graph_builder.add_node("data_collection", data_collection)
    graph_builder.add_node("data_conversion", data_conversion)

    graph_builder.add_edge(START, "data_collection")
    graph_builder.add_edge("data_collection", "data_conversion")
    graph_builder.add_edge("data_conversion", END)

    graph = graph_builder.compile()
    final_state: DataCollectionState = await graph.ainvoke(state)
    print("Final State:", final_state)



if __name__ == "__main__":
    asyncio.run(main())

