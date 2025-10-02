from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager


class RefineTargetAnalyzer(BaseAgent):
    @property
    def role_name(self) -> str:
        return "refine_target_analyzer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_refine_target_analyzer"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_refine_target_analyzer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "purpose": pre_tool_results.get("purpose", ""),
            "pipeline_nodes_summary": pre_tool_results.get("pipeline_nodes_summary", []),
            # 传入完整的 pipeline JSON（字符串形式），帮助模型避免误判“空”
            "pipeline_code": pre_tool_results.get("get_pipeline_code", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"purpose": "", "pipeline_nodes_summary": []}


class RefinePlanner(BaseAgent):
    @property
    def role_name(self) -> str:
        return "refine_planner"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_refine_planner"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_refine_planner"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "intent": pre_tool_results.get("intent", {}),
            "pipeline_nodes_summary": pre_tool_results.get("pipeline_nodes_summary", []),
            "op_context": pre_tool_results.get("op_context", {}),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"intent": {}, "pipeline_nodes_summary": [], "op_context": {}}


class JsonPipelineRefiner(BaseAgent):
    @property
    def role_name(self) -> str:
        return "pipeline_refiner"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_json_pipeline_refiner"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_json_pipeline_refiner"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_json": pre_tool_results.get("pipeline_json", {}),
            "modification_plan": pre_tool_results.get("modification_plan", {}),
            "op_context": pre_tool_results.get("op_context", {}),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"pipeline_json": {}, "modification_plan": {}, "op_context": {}}


def create_refine_target_analyzer(tool_manager: Optional[ToolManager] = None, **kwargs) -> RefineTargetAnalyzer:
    return RefineTargetAnalyzer(tool_manager=tool_manager or get_tool_manager(), **kwargs)


def create_refine_planner(tool_manager: Optional[ToolManager] = None, **kwargs) -> RefinePlanner:
    return RefinePlanner(tool_manager=tool_manager or get_tool_manager(), **kwargs)


def create_json_pipeline_refiner(tool_manager: Optional[ToolManager] = None, **kwargs) -> JsonPipelineRefiner:
    return JsonPipelineRefiner(tool_manager=tool_manager or get_tool_manager(), **kwargs)
