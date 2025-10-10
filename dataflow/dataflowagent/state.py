from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict
from dataflow.cli_funcs.paths import DataFlowPath
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
STATICS_DIR = DataFlowPath.get_dataflow_statics_dir()

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

@dataclass
class DFRequest:
    # ① 用户偏好的自然语言
    language: str = "en"                          # "en" | "zh" | ...

    # ②  LLM 接口
    chat_api_url: str = "http://123.129.219.111:3000/v1"   # OpenAI 兼容接口地址
    api_key: str = os.getenv("DF_API_KEY", "test")         # 默认从环境变量读取

    # ③ 选用的 LLM 名称
    model: str = "gpt-4o"                       # 例如 "gpt-4o" / "gpt-3.5-turbo" / 自托管模型名

    # ④ 测试样例文件（仅 CLI 批量跑用）
    #    这里默认放在项目路径 dataflow/example/DataflowAgent/mq_test_data.jsonl
    json_file: str = (
        f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    )

    # ⑤ 需求，算子或者pipeline；
    target: str = ""                             

    # ⑥ 让 Agent 读取 / 生成 Python 代码时使用的文件位置
    python_file_path: str = ""                   # 为空表示自动分配临时目录

    # ⑦ 是否进入 Debug 模式
    need_debug: bool = False
    max_debug_rounds: int = 3                    # 最多 Debug 几次

    # ⑧ 是否使用本地模型而非远程 OpenAI
    use_local_model: bool = False
    local_model_path: str = ""                   # 指向 .gguf / .bin / huggingface checkpoint


@dataclass
class DFState:
    request: DFRequest
    messages: Annotated[list[BaseMessage], add_messages]
    agent_results: Dict[str, Any] = field(default_factory=dict)
    category: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict) # 供 Agent 之间传递临时数据，不伴随整个生命周期，可以随时clear；
    # 匹配阶段产出的候选算子名单（用于 writer 上下文注入与后续复用）
    matched_ops: list[str] = field(default_factory=list)
    debug_mode: bool = False
    # 当前流水线的结构JSON（包含nodes与edges），供精修/修改流程读取与写回
    pipeline_structure_code: Dict[str, Any] = field(default_factory=dict)
    
    execution_result: Dict[str, Any] = field(default_factory=dict)
    code_debug_result: Dict[str, Any] = field(default_factory=dict)
    # structure_code : Dict[str, Any] = field(default_factory=dict)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class DataCollectionRequest:
    language: str = "English"
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")
    model: str = "gpt-4o"
    target: str = ""
    download_dir: str = os.path.join(STATICS_DIR, "data_collection")
    dataset_size_category: str = '1K<n<10K'
    dataset_num_limit: int = 5
    category: str = "PT"

@dataclass
class DataCollectionState:
    request: DataCollectionRequest
    keywords: list[str] = field(default_factory=list)
    datasets: Dict[str, list] = field(default_factory=dict)
    downloads: Dict[str, list] = field(default_factory=dict)
    sources: Dict[str, Dict] = field(default_factory=dict)