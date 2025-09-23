from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dataflow import get_logger
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager

log = get_logger()

class Rewriter(BaseAgent):
    # ---------------- BaseAgent 元数据 ----------------
    @property
    def role_name(self) -> str:
        return "rewriter"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_code_rewriting"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_code_rewriting"

    # ---------------- Prompt 参数 --------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "error_trace": pre_tool_results.get("error_trace", ""),
            "debug_reason": pre_tool_results.get("debug_reason", ""),
            "data_sample": pre_tool_results.get("data_sample", ""),
        }

    # ---------------- 默认值 -------------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "pipeline_code": "",
            "error_trace": "",
            "debug_reason": "",
            "data_sample": "",
        }

    # ---------------- 结果落盘 -----------------------
    def _dump_new_code(self, state: DFState, new_code: str) -> Path | None:
        """
        将新代码写回原 python 文件。
        默认使用 state.execution_result["file_path"]，
        若不存在则写入 ./tmp_rewrite.py 供人工检查。
        """
        file_path_str: str | None = None
        if isinstance(state.execution_result, dict):
            file_path_str = state.execution_result.get("file_path")  # 由 PipelineBuilder 产出
        # 允许 caller 提前把目标文件路径放进 temp_data
        file_path_str = file_path_str or state.temp_data.get("pipeline_file_path")

        if not file_path_str:
            log.warning("无法确定 Pipeline 文件路径，已保存到临时文件 ./tmp_rewrite.py")
            file_path = Path("tmp_rewrite.py").resolve()
        else:
            file_path = Path(file_path_str).resolve()

        file_path.write_text(new_code, encoding="utf-8")
        return file_path

    # ---------------- 更新 DFState -------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：
            {
              "code":   "..."
            }
        """
        new_code: str = result.get("code", "")
        if new_code:
            saved_path = self._dump_new_code(state, new_code)
            log.info(f"[rewriter] 新代码已保存到 {saved_path}")
            state.temp_data["pipeline_code"] = new_code
            state.temp_data["pipeline_file_path"] = str(saved_path)

        state.rewrite_result = result

        super().update_state_result(state, result, pre_tool_results)

    def after_rewrite(self, state: DFState) -> DFState:
        """
        供 LangGraph 使用的辅助方法：
        - 轮次计数 +1
        - 标记本轮已经重写
        """
        state.temp_data["round"] = state.temp_data.get("round", 0) + 1
        state.temp_data["rewritten"] = True
        return state


async def rewrite_code(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    rewriter = Rewriter(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await rewriter.execute(state, use_agent=use_agent, **kwargs)

def create_rewriter(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> Rewriter:
    return Rewriter(tool_manager=tool_manager, **kwargs)