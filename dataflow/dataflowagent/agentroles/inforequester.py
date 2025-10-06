from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataflow import get_logger
from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState

log = get_logger()


class InfoRequester(BaseAgent):
    """
    询问信息-Agent  
    1. 读取 pipeline_code + error_trace  
    2. 让 LLM 判断还需要哪些模块源码 → **必须调用工具 `get_otherinfo_code`**  
    3. 工具返回源码后，LLM 最终产出  
       { "other_info": "对源码内容的综合说明 / 关键信息摘要" }
    """

    # ---------- 元数据 ----------
    @property
    def role_name(self) -> str:
        return "info_requester"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_other_info_request"

    @property
    def task_prompt_template_name(self) -> str:
        # 模板名称已更改
        return "task_prompt_for_context_collection"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(
        self, pre_tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "pipeline_code": pre_tool_results.get("pipeline_code", ""),
            "error_trace": pre_tool_results.get("error_trace", ""),
        }

    # ---------- 默认前置工具 ----------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"pipeline_code": "", "error_trace": ""}

    # ---------- 结果解析 ----------
    def parse_result(self, content: str) -> dict:
        """
        只关心 other_info，解析失败时把全文当作 other_info。
        不调用 BaseAgent.parse_result()，避免无意义的 WARNING。
        """
        import json, re

        # ① 尝试按整段 JSON 解析
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "other_info" in data:
                return {"other_info": data["other_info"]}
        except Exception:
            pass

        # ② 正则抠第一段 {...}
        m = re.search(r"\{[\s\S]*?\}", content)
        if m:
            try:
                data = json.loads(m.group())
                if "other_info" in data:
                    return {"other_info": data["other_info"]}
            except Exception:
                pass

        # ③ 兜底
        return {"other_info": content.strip()}

    # ---------- 写回 DFState ----------
    def update_state_result(
        self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]
    ):
        print("result::::::",result)
        summary: str = result.get("other_info", "")
        log.info(f'[InfoRequester result]: {result}')
        state.temp_data["other_info_summary"] = summary
        super().update_state_result(state, result, pre_tool_results)


# ---------- 工厂 / helper ----------
def create_info_requester(tool_manager=None, **kwargs) -> InfoRequester:
    return InfoRequester(tool_manager=tool_manager, **kwargs)


async def request_other_info(
    state: DFState,
    tool_manager=None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = True,
) -> DFState:
    agent = InfoRequester(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await agent.execute(state, use_agent=use_agent)