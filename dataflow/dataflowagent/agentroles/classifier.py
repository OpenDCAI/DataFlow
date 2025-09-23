from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow.dataflowagent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.utils import robust_parse_json
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()

from .base_agent import BaseAgent

class DataContentClassifier(BaseAgent):
    """数据内容分类器 - 继承自BaseAgent"""
    
    @property
    def role_name(self) -> str:
        return "classifier"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_content_classification"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_content_classification"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """数据分类器特有的提示词参数"""
        return {
            'local_tool_for_sample': pre_tool_results.get('sample', ''),
            'local_tool_for_get_categories': pre_tool_results.get('categories', '[]'),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """数据分类器的默认前置工具结果"""
        return {
            'sample': '',
            'categories': '[]'
        }
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 保持向后兼容"""
        state.category = result 
        super().update_state_result(state, result, pre_tool_results)

async def data_content_classification(
    state: DFState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    classifier = DataContentClassifier(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await classifier.execute(state, use_agent=use_agent, **kwargs)

def create_classifier(tool_manager: Optional[ToolManager] = None, **kwargs) -> DataContentClassifier:
    return DataContentClassifier(tool_manager=tool_manager, **kwargs)