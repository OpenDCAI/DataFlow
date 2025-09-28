# dataflow/dataflowagent/agentroles/recommender.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager

from dataflow import get_logger
log = get_logger()


class DataPipelineRecommender(BaseAgent):
    """
    根据样本、目标意图和算子库，生成数据处理管线推荐。
    前置工具:
        - sample   : local_tool_for_sample
        - target   : local_tool_for_get_purpose
        - operator : get_operator_content_map_from_all_operators
    后置工具:
        - post_process_combine_pipeline_result  (LangChain Tool)
    """
    @property
    def role_name(self) -> str:
        return "recommender"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_recommendation_inference_pipeline"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_recommendation_inference_pipeline"

    # --- 向任务提示词中注入变量 --------------------------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 变量名。
        这些变量名要与模板文件里的占位符一致。
        """
        return {
            "sample": pre_tool_results.get("sample", ""),
            "target": pre_tool_results.get("target", ""),
            "operator": pre_tool_results.get("operator", "[]"),
        }

    # --- 默认前置工具结果（兜底）-------------------------------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "sample": "",
            "target": "",
            "operator": "[]",
        }

    # --- 将结果写回 DFState -------------------------------------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        # 对外暴露属性名为 recommendation
        state.recommendation = result
        super().update_state_result(state, result, pre_tool_results)

    def create_assistant_node_func(self, state, pre_tool_results):
        async def assistant_node(graph_state: DFState):
            msgs = getattr(graph_state, "messages", [])
            if not msgs:
                msgs = self.build_messages(graph_state, pre_tool_results)

            resp = await self.process_with_llm_for_graph(msgs, graph_state)
            if self.has_tool_calls(resp):
                log.info(f"recommender: LLM选择调用工具: {[call for call in resp.tool_calls]}")
                state.messages= msgs + [resp]
                return {"messages": msgs + [resp]}        # 让工具节点去执行
            else:
                result = self.parse_result(resp.content)
                self.update_state_result(graph_state, result, pre_tool_results)
                state.messages= msgs + [resp]
                state.recommendation = result
                return {
                    "messages": msgs + [resp],
                    "recommendation": result,              
                    "finished": True,
                }
        return assistant_node

async def data_pipeline_recommendation(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    recommender = DataPipelineRecommender(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await recommender.execute(state, use_agent=use_agent, **kwargs)


def create_recommender(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DataPipelineRecommender:
    return DataPipelineRecommender(tool_manager=tool_manager, **kwargs)