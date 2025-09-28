"""
DataPipelineBuilder
~~~~~~~~~~~~~~~~~~~
1) 根据推荐算子列表调用 pipeline_assembler 生成 python 代码
2) 落盘为 .py 文件
3) 启动子进程执行，捕获运行结果

扩展：
    - skip_assemble=True  : 仅执行现有脚本而不重新组装
    - file_path           : 指定要执行的脚本路径
支持调试模式：state.debug_mode = True 时仅取前 10 行数据加速调试。
"""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataflow import get_logger
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow.dataflowagent.toolkits.pipetool.pipe_tools import (
    pipeline_assembler,
    write_pipeline_file,
)

log = get_logger()

# ---------------------------------------------------------------------- #
#                              工具函数                                   #
# ---------------------------------------------------------------------- #

def _patch_first_entry_file(py_file: str | Path,
                            old_path: str,
                            new_path: str) -> None:
    """
    把脚本中的 first_entry_file_name 由 old_path 替换成 new_path
    """
    py_file = Path(py_file).expanduser().resolve()
    code = py_file.read_text(encoding="utf-8")

    # 既考虑单/双引号，也兼容额外空格
    pattern = (
        r'first_entry_file_name\s*=\s*[\'"]'
        + re.escape(old_path)
        + r'[\'"]'
    )
    replacement = f'first_entry_file_name=\"{new_path}\"'
    new_code, n = re.subn(pattern, replacement, code, count=1)
    if n == 0:
        # 保险：直接字符串替换
        new_code = code.replace(old_path, new_path)

    py_file.write_text(new_code, encoding="utf-8")

def _ensure_py_file(code: str, file_name: str | None = None) -> Path:
    """
    把生成的代码写入文件并返回路径。
    若 file_name 为空，写入系统临时目录。
    """
    if file_name:
        target = Path(file_name).expanduser().resolve()
    else:
        target = Path(tempfile.gettempdir()) / f"recommend_pipeline_{uuid.uuid4().hex}.py"
    target.write_text(textwrap.dedent(code), encoding="utf-8")
    log.warning(f"[pipeline_builder] pipeline code written to {target}")
    return target


def _create_debug_sample(src_file: str | Path, sample_lines: int = 10) -> Path:
    """
    从 src_file 抽取前 sample_lines 行，写入临时文件并返回路径。
    """
    src_path = Path(src_file).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source file not found: {src_path}")

    tmp_path = (
        Path(tempfile.gettempdir())
        / f"{src_path.stem}_sample_{sample_lines}{src_path.suffix}"
    )

    with src_path.open("r", encoding="utf-8") as rf, tmp_path.open(
        "w", encoding="utf-8"
    ) as wf:
        for idx, line in enumerate(rf):
            if idx >= sample_lines:
                break
            wf.write(line)

    log.info(
        f"[pipeline_builder] debug mode: sample data written to {tmp_path} "
        f"(first {sample_lines} lines)"
    )
    return tmp_path


async def _run_py(file_path: Path) -> Dict[str, Any]:
    """异步执行 python 文件并捕获输出"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "success": proc.returncode == 0,
        "return_code": proc.returncode,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "file_path": str(file_path),
    }


# ---------------------------------------------------------------------- #
#                                Agent                                    #
# ---------------------------------------------------------------------- #
class DataPipelineBuilder(BaseAgent):
    """把推荐算子列表转换为完整 Pipeline 并立即执行，支持调试与只执行两种模式"""

    # ---------- 基本信息 ----------
    @property
    def role_name(self) -> str:
        return "pipeline_builder"

    # 本 Agent 不调用 LLM，模板仅作占位
    @property
    def system_prompt_template_name(self) -> str:  # noqa: D401
        return "VOID"

    @property
    def task_prompt_template_name(self) -> str:
        return "VOID"

    # ----------- 前置工具默认结果 ------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"recommendation": []}

    # ----------- 主执行逻辑 ------------------
    async def execute(
        self,
        state: DFState,
        *,
        skip_assemble: bool = False,
        file_path: str | None = None,
        assembler_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> DFState:  # type: ignore[override]
        """
        运行模式说明
        ------------------------------------------------------------------
        1. skip_assemble=False (默认) : 正常「组装→写盘→执行」全流程
        2. skip_assemble=True        : 仅执行已存在的脚本 file_path
                                       (若 file_path 为空则取 self.temp_data['pipeline_file_path'])
        ------------------------------------------------------------------
        """
        # assembler_kwargs = assembler_kwargs or {}
        assembler_kwargs = dict(assembler_kwargs or {})
        try:
            # ---------------- ① 需要重新组装代码 -----------------
            if not skip_assemble:
                # 1) 获取推荐算子
                pre_tool_results = await self.execute_pre_tools(state)
                recommendation: List[str] = (
                    pre_tool_results.get("recommendation")
                    or getattr(state, "recommendation", [])
                )
                if not recommendation:
                    raise ValueError("无可用 recommendation")

                # -------- 调试模式处理 --------
                debug_mode: bool = bool(getattr(state, "debug_mode", False))
                if debug_mode:
                    origin_file: str | None = assembler_kwargs.get("file_path")
                    if not origin_file:
                        raise ValueError(
                            "debug 模式下需要 `assembler_kwargs['file_path']` 指向原始数据文件"
                        )
                    sample_path = _create_debug_sample(origin_file, sample_lines=10)
                    assembler_kwargs["file_path"] = str(sample_path)
                    state.temp_data["debug_sample_file"] = str(sample_path)
                    state.temp_data["origin_file_path"] = origin_file
                    log.info(f"[pipeline_builder] DEBUG mode , sample at {sample_path}")

                # 2) 生成 pipeline 代码字符串
                pipe_obj = pipeline_assembler(recommendation, **assembler_kwargs)
                print(f"assembler_kwargs : {assembler_kwargs}")
                code_str: str = pipe_obj["pipe_code"]

                # 记录代码到状态
                state.pipeline_code = code_str
                state.temp_data["pipeline_code"] = code_str

                # 3) 写临时代码文件
                file_path_obj = _ensure_py_file(code_str, file_name=file_path)
                state.pipeline_file_path = str(file_path_obj)
                file_path = str(file_path_obj)  # 供后续 _run_py 使用

            # ---------------- ② 仅执行已存在脚本 -----------------
            else:
                file_path = file_path or state.temp_data.get("pipeline_file_path")
                if not file_path:
                    raise ValueError("skip_assemble=True 但未提供 file_path")
                file_path_obj = Path(file_path).expanduser().resolve()
                if not file_path_obj.is_file():
                    raise FileNotFoundError(f"待执行文件不存在: {file_path_obj}")

            # ---------------- ③ 真执行 -----------------------
            exec_result = await _run_py(Path(file_path))
            state.execution_result = exec_result
            log.info(f"[pipeline_builder] run success={exec_result['success']}")

            # 若调试成功，关闭 debug 开关，以便后续跑全量数据
            if getattr(state, "debug_mode", False) and exec_result["success"]:
                state.debug_mode = False
                log.info("[pipeline_builder] debug run passed, state.debug_mode -> False")

                sample_path: str | None = state.temp_data.pop("debug_sample_file", None)
                origin_path: str | None = state.temp_data.get("origin_file_path")
                if sample_path and origin_path:
                    _patch_first_entry_file(
                        py_file=state.request.python_file_path,   # 调试期生成的脚本
                        old_path=sample_path,
                        new_path=origin_path,
                    )
                    log.info(f"[pipeline_builder] patched first_entry_file_name -> {origin_path}")
                    # exec_result = await _run_py(Path(state.pipeline_file_path))
                    # state.execution_result = exec_result
                    log.info(f"[pipeline_builder] full run success={exec_result['success']}")

        except Exception as e:
            log.exception("[pipeline_builder] 构建/执行失败")
            state.execution_result = {
                "success": False,
                "stderr": str(e),
                "stdout": "",
                "return_code": -1,
            }
            
        self.update_state_result(state, state.execution_result, locals().get("pre_tool_results", {}))  # type: ignore[arg-type]
        return state


# ---------------------------------------------------------------------- #
#                    对外统一调用入口                                     #
# ---------------------------------------------------------------------- #
async def data_pipeline_build(
    state: DFState,
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DFState:
    """单步调用：构建并执行推荐管线"""
    builder = DataPipelineBuilder(tool_manager=tool_manager)
    return await builder.execute(state, **kwargs)


def create_pipeline_builder(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DataPipelineBuilder:
    return DataPipelineBuilder(tool_manager=tool_manager, **kwargs)