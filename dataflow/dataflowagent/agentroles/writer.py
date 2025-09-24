from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

from .base_agent import BaseAgent

log = get_logger()


class Writer(BaseAgent):
    @property
    def role_name(self) -> str:
        # 与 prompts_repo 中“写算子”模板键匹配
        return "write_the_operator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_write_the_operator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_write_the_operator"

    # ---------------- Prompt 参数 --------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "example": pre_tool_results.get("example", ""),
            "target": pre_tool_results.get("target", ""),
        }

    # ---------------- 默认值 -------------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "example": "",
            "target": "",
        }

    def _dump_code(self, state: DFState, new_code: str) -> Path | None:
        """
        将新代码写入目标文件。如果未提供路径，则仅返回 None（不强制落盘）。
        优先顺序：
          1) state.execution_result["file_path"]
          2) state.temp_data["pipeline_file_path"]
        """
        file_path_str: str | None = None
        if isinstance(state.execution_result, dict):
            file_path_str = state.execution_result.get("file_path")
        file_path_str = file_path_str or state.temp_data.get("pipeline_file_path")

        if not file_path_str:
            # 不强制写入临时文件，避免误写；仅提示
            log.info("未提供目标文件路径，跳过写盘（代码保存在 state.draft_operator_code）")
            return None

        file_path = Path(file_path_str)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            log.info(f"已将新代码写入 {file_path}")
            return file_path
        except Exception as e:
            log.error(f"写入文件 {file_path} 失败: {e}")
            return None

    # ---------------- 更新 DFState -------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        code_str = ""
        if isinstance(result, dict):
            code_str = result.get("code", "")
        state.draft_operator_code = code_str
        if code_str:
            self._dump_code(state, code_str)
        super().update_state_result(state, result, pre_tool_results)


async def code_writing(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    inst = create_writer(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_writer(tool_manager: Optional[ToolManager] = None, **kwargs) -> Writer:
    if tool_manager is None:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return Writer(tool_manager=tool_manager, **kwargs)
