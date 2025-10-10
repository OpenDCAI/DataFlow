from __future__ import annotations

import tempfile
import uuid
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

from dataflow import get_logger
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow.dataflowagent.agentroles.pipelinebuilder import _run_py

log = get_logger()


def _ensure_py_file(code: str, file_name: str | None = None) -> Path:
    """
    将代码写入文件并返回路径。
    若 file_name 为空，写入系统临时目录。
    """
    if file_name:
        target = Path(file_name).expanduser().resolve()
    else:
        target = Path(tempfile.gettempdir()) / f"operator_{uuid.uuid4().hex}.py"
    target.write_text(textwrap.dedent(code), encoding="utf-8")
    log.warning(f"[operator_executor] code written to {target}")
    return target


class OperatorExecutor(BaseAgent):
    """执行单个算子代码（不调用LLM）"""

    @property
    def role_name(self) -> str:
        return "operator_executor"

    @property
    def system_prompt_template_name(self) -> str:  # noqa: D401
        return "VOID"

    @property
    def task_prompt_template_name(self) -> str:
        return "VOID"

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    async def execute(
        self,
        state: DFState,
        *,
        file_path: str | None = None,
        **kwargs,
    ) -> DFState:  # type: ignore[override]
        try:
            pre_tool_results = await self.execute_pre_tools(state)

            code_str: str = (
                state.temp_data.get("pipeline_code")
                or getattr(state, "draft_operator_code", "")
            )
            if not code_str:
                raise ValueError(
                    "无可执行的算子代码：draft_operator_code / temp_data['pipeline_code'] 为空"
                )

            # 写入目标文件
            file_path = file_path or state.temp_data.get("pipeline_file_path")
            file_path_obj = _ensure_py_file(code_str, file_name=file_path)
            state.temp_data["pipeline_file_path"] = str(file_path_obj)

            # 执行并捕获结果
            exec_result = await _run_py(file_path_obj)
            state.execution_result = exec_result
            log.info(f"[operator_executor] run success={exec_result['success']}")

        except Exception as e:
            log.exception("[operator_executor] 执行失败")
            state.execution_result = {
                "success": False,
                "stderr": str(e),
                "stdout": "",
                "return_code": -1,
            }

        self.update_state_result(state, state.execution_result, locals().get("pre_tool_results", {}))  # type: ignore[arg-type]
        return state


async def operator_execute(
    state: DFState,
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DFState:
    inst = OperatorExecutor(tool_manager=tool_manager)
    return await inst.execute(state, **kwargs)


def create_operator_executor(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> OperatorExecutor:
    return OperatorExecutor(tool_manager=tool_manager, **kwargs)

