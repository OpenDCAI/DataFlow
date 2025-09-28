# dataflow/dataflowagent/agentroles/code_debugger.py
# -*- coding: utf-8 -*-
"""
CodeDebugger —— 捕获并分析执行管线时报错的调试 Agent

前置工具（可选，按需配置 ToolManager）：
    - pipeline_code : 由 PipelineBuilder / RewriteAgent 生成的最新管线代码字符串
    - error_trace   : ExecuteAgent 捕获的异常堆栈

后置工具（可选，例如让 LLM 调工具自动修改代码）：
    - fix_tool      : 自定义 Tool，把 LLM 给出的补丁应用到文件

本 Agent 仅负责：
    1. 读取 “pipeline_code + error_trace” 两段上下文；
    2. 让 LLM 给出『调试分析 + 详细修改建议』的 JSON 结果。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()


class CodeDebugger(BaseAgent):
    @property
    def role_name(self) -> str:
        return "code_debugger"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_code_debugging"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_code_debugging"

    # -------------------- Prompt 参数 -------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 中的占位符：
            {{ pipeline_code }}   – 需要调试的代码
            {{ error_trace }}     – 本次执行捕获的异常信息
        """
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "error_trace": pre_tool_results.get("error_trace", ""),
        }

    # -------------------- 前置工具默认值 -----------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "pipeline_code": "",
            "error_trace": "",
        }

    # -------------------- 结果写回 DFState --------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：
            reason: str      – 调试分析
        """
        state.code_debug_result = result
        super().update_state_result(state, result, pre_tool_results)


# ------------------------------------------------------------------
#                    对外统一调用入口（函数封装）
# ------------------------------------------------------------------
async def code_debug(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """
    单步调用：执行 CodeDebugger 并将结果写回 DFState
    """
    debugger = CodeDebugger(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await debugger.execute(state, use_agent=use_agent, **kwargs)


def create_code_debugger(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> CodeDebugger:
    return CodeDebugger(tool_manager=tool_manager, **kwargs)