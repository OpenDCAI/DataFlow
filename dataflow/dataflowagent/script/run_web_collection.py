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
        target = "我需要一些金融和法律数据",    ## 输入的自然语言指令，可以包括多个领域关键词
        category = "SFT",   ## 需要的数据类别，可选值为PT/SFT。PT表示预训练数据，SFT表示指令微调数据
        dataset_num_limit = 5,  ## 用于hf搜索，每个关键词对应的hf数据集数量上限
        dataset_size_category = '1K<n<10K'   ## 用于hf搜索，每个hf数据集的样本数范围。可选值包括 'n<1K', '1K<n<10K', '10K<n<100K', '100K<n<1M', 'n>1M'
        ## download_dir  下载的数据位置。默认为os.path.join(STATICS_DIR, "data_collection")，每个关键词一个子目录，包含tmp目录(原始hf数据集)和jsonl文件(处理后的数据文件)
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



if __name__ == "__main__":
    asyncio.run(main())

