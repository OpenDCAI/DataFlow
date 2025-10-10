from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager


class InstantiateAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "llm_instantiate"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_llm_instantiate"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_llm_instantiate"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": pre_tool_results.get("target", ""),
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "example_data": pre_tool_results.get("example_data", []),
            "available_keys": pre_tool_results.get("available_keys", []),
            "preselected_input_key": pre_tool_results.get("preselected_input_key", ""),
            "test_data_path": pre_tool_results.get("test_data_path", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "target": "",
            "pipeline_code": "",
            "example_data": [],
            "available_keys": [],
            "preselected_input_key": "",
            "test_data_path": "",
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


def create_llm_instantiator(tool_manager: Optional[ToolManager] = None, **kwargs) -> InstantiateAgent:
    if tool_manager is None:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return InstantiateAgent(tool_manager=tool_manager, **kwargs)
