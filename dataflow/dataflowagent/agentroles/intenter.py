# dataflow/dataflowagent/agentroles/intenter.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from dataflow import get_logger
from dataflow.dataflowagent.promptstemplates.prompt_template import (
    PromptsTemplateGenerator,
)
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow.dataflowagent.utils import robust_parse_json

# 统一日志实例
log = get_logger()

# 复用公共父类
from .base_agent import BaseAgent


class Intenter(BaseAgent):
    """
    意图理解 Agent
    --------------------------------------------------
    任务：分析当前对话 / 输入，判断用户想完成的意图
    结果：写入 DFState.intent，结构示例：
        {
            "intent": "query_table",
            "arguments": {
                "table_name": "sales_2024",
                "goal": "top10"
            }
        }
    """

    # ============= BaseAgent 必备接口 =============

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        """
        工厂方法，保持与其它 Agent 的一致调用方式
        """
        return cls(tool_manager=tool_manager, **kwargs)

    # —— 给 BaseAgent 用的 meta 属性 ——
    @property
    def role_name(self) -> str:
        return "intenter"

    @property
    def system_prompt_template_name(self) -> str:
        """
        提示词模板名称，需在 promptstemplates/ 同名 yml 或 txt 中实现
        """
        return "system_prompt_for_intent_recognition"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_intent_recognition"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intenter 只需要用户最新一条消息，如有额外信息可在这里注入
        """
        return {
            "latest_user_message": pre_tool_results.get("latest_user_message", ""),
            "conversation_excerpt": pre_tool_results.get("conversation_excerpt", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """
        若未配置外部工具，这里提供兜底字段，防止 prompt 渲染 KeyError
        """
        return {
            "latest_user_message": "",
            "conversation_excerpt": "",
        }

    def update_state_result(
        self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]
    ):
        """
        把解析后的意图写入 state.intent；其余交由父类处理（记录 agent_results 等）
        """
        state.intent = result
        super().update_state_result(state, result, pre_tool_results)


# --------------------------------------------------
#             对外的快捷函数封装
# --------------------------------------------------
async def intent_recognition(
    state: DFState,
    *,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """
    直接调用该函数即可完成一次意图识别并返回新的 state
    """
    intenter = Intenter(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await intenter.execute(state, use_agent=use_agent, **kwargs)


def create_intenter(tool_manager: Optional[ToolManager] = None, **kwargs) -> Intenter:
    """
    保持与其它 *create_xxx* API 一致的对象构造器
    """
    return Intenter(tool_manager=tool_manager, **kwargs)