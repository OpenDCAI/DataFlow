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
        # 维持模板仅使用 {example},{target}；将数据上下文并入 example
        example = pre_tool_results.get("example", "")
        data_sample = pre_tool_results.get("data_sample", [])
        available_keys = pre_tool_results.get("available_keys", [])
        try:
            import json
            preview = (
                "\n\n# 数据样例预览\n"
                + json.dumps(data_sample[:2], ensure_ascii=False)
                + "\n# 数据中可用的字段（keys）\n"
                + json.dumps(available_keys, ensure_ascii=False)
                + "\n# 运行约束\n"
                + (
                    "调试阶段不会传入 input_key。请将算子运行接口定义为 "
                    "run(self, storage: DataFlowStorage, input_key: str | None = None, output_key: 根据实际情况选择)。"
                    "当 input_key 为 None 时，算子在内部必须自动选择合适的输入字段，不允许因为缺少 input_key 而报错。"
                )
                + "\n# 自动选择策略（多层兜底）\n"
                + "1) 名称优先：raw_content、text、content、sentence、instruction、input、query、problem、prompt。\n"
                + "2) 类型统计打分：按字符串占比、非空占比、平均长度、唯一率综合评分选择文本列。\n"
                + "3) LLM建议仅作 hint：结合 target、样例与 available_keys 让模型建议字段，最终需校验在 available_keys 且非空。\n"
                + "4) 保守兜底：若仍无法确定，选择最文本化的列；再不行，取第一列转字符串并记录低置信度日志。\n"
                + "\n# 输出约定\n"
                + "输出字段名使用 output_key，默认值请设置为任务合理的名称。\n"
            )
            example = (example or "") + preview
        except Exception:
            pass
        return {
            "example": example,
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
        # 将生成代码写入状态，并同步到 temp_data 以便后续执行/调试节点复用
        state.draft_operator_code = code_str
        if code_str:
            saved_path = self._dump_code(state, code_str)
            try:
                state.temp_data["pipeline_code"] = code_str
                if saved_path is not None:
                    state.temp_data["pipeline_file_path"] = str(saved_path)
            except Exception:
                pass
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
