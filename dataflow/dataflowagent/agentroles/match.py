from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

from .base_agent import BaseAgent

log = get_logger()


class MatchOperator(BaseAgent):
    @property
    def role_name(self) -> str:
        # 与 prompts_repo 中的模板键保持一致
        return "match_operator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_match_operator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_match_operator"

    # ---------------- Prompt 参数 --------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "get_operator_content": pre_tool_results.get("get_operator_content", ""),
            "purpose": pre_tool_results.get("purpose", ""),
        }

    # ---------------- 默认值 -------------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "get_operator_content": "",
            "purpose": "",
        }

    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        # 兼容：将匹配到的算子名列表同步到 DFState.matched_ops，供后续 writer 使用
        try:
            matched = result.get("match_operators", []) if isinstance(result, dict) else []
            if isinstance(matched, list):
                state.matched_ops = matched
        except Exception:
            pass
        return super().update_state_result(state, result, pre_tool_results)


async def match_operator(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    inst = create_match(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_match(tool_manager: Optional[ToolManager] = None, **kwargs) -> MatchOperator:
    if tool_manager is None:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return MatchOperator(tool_manager=tool_manager, **kwargs)
