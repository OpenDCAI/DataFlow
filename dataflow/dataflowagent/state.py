from dataclasses import dataclass, field
import os
from typing import Any, Dict
from dataflow.cli_funcs.paths import DataFlowPath
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
@dataclass
class DFRequest:
    language: str = "en"
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")
    model: str = "gpt-4o"
    json_file: str = f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl" 
    target: str = ""
    python_file_path: str = ""
    need_debug: bool = False
    max_debug_rounds: int = 3


@dataclass
class DFState:
    request: DFRequest
    messages: Annotated[list[BaseMessage], add_messages] # cnm 一定要是追加，2025年9月19日03:29:13
    agent_results: Dict[str, Any] = field(default_factory=dict)
    category: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    temp_data: Dict[str, Any] = field(default_factory=dict) # 供 Agent 之间传递临时数据，不伴随整个生命周期，可以随时clear；
    debug_mode: bool = True
    execution_result: Dict[str, Any] = field(default_factory=dict)
    code_debug_result: Dict[str, Any] = field(default_factory=dict)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __setitem__(self, key, value):
        setattr(self, key, value)