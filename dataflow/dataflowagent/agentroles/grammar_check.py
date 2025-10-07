from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow import get_logger
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager

log = get_logger()


class GrammarChecker(BaseAgent):
    @property
    def role_name(self) -> str:
        return "grammar_check"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_grammar_check"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_grammar_check"

    # -------- Prompt 参数 --------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "sample_data": pre_tool_results.get("sample_data", []),
            "available_keys": pre_tool_results.get("available_keys", []),
            "target": pre_tool_results.get("target", ""),
        }

    # -------- 默认前置工具结果 --------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "pipeline_code": "",
            "sample_data": [],
            "available_keys": [],
            "target": "",
        }

    # -------- 结果写回 DFState --------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        # 期望结果：{"grammar_ok": bool, "message": str, "fixed_code": str?}
        grammar_ok = bool(result.get("grammar_ok", False))
        message = result.get("message", "") or ""
        fixed_code = result.get("fixed_code", "") or ""

        # 应用轻量修复（仅内存，不落盘）
        if fixed_code:
            try:
                state.temp_data["pipeline_code"] = fixed_code
                state.draft_operator_code = fixed_code
                log.info("[grammar_check] 应用 fixed_code 至内存（不落盘）")
            except Exception:
                pass

        # 将 grammar 错误写入 execution_result 便于 rewriter 读取 error_trace
        try:
            if not grammar_ok and message:
                state.execution_result.setdefault("stdout", "")
                state.execution_result["stderr"] = message
                state.execution_result.setdefault("traceback", message)
        except Exception:
            pass

        # 默认写回 agent_results
        super().update_state_result(state, result, pre_tool_results)


async def run_grammar_check(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    agent = GrammarChecker(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_grammar_checker(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> GrammarChecker:
    return GrammarChecker(tool_manager=tool_manager, **kwargs)

