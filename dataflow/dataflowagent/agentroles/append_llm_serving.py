from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager


class AppendLLMServingAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "llm_append_serving"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_llm_append_serving"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_llm_append_serving"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "llm_serving_snippet": pre_tool_results.get("llm_serving_snippet", ""),
            "example_data": pre_tool_results.get("example_data", []),
            "available_keys": pre_tool_results.get("available_keys", []),
            "target": pre_tool_results.get("target", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "pipeline_code": "",
            "llm_serving_snippet": "",
            "example_data": [],
            "available_keys": [],
            "target": "",
        }

    def _dump_code(self, state: DFState, code: str) -> Optional[Path]:
        file_path = state.temp_data.get("pipeline_file_path") or getattr(state.request, "python_file_path", "")
        if not file_path:
            return None
        p = Path(file_path)
        try:
            p.write_text(code, encoding="utf-8")
            return p
        except Exception:
            return None

    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        code = result.get("code", "") if isinstance(result, dict) else ""
        if code:
            state.temp_data["pipeline_code"] = code
            state.draft_operator_code = code
            saved = self._dump_code(state, code)
            if saved:
                state.temp_data["pipeline_file_path"] = str(saved)
        super().update_state_result(state, result, pre_tool_results)


def create_llm_append_serving(tool_manager: Optional[ToolManager] = None, **kwargs) -> AppendLLMServingAgent:
    if tool_manager is None:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return AppendLLMServingAgent(tool_manager=tool_manager, **kwargs)
