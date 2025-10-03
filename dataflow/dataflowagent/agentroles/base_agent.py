from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow.dataflowagent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.utils import robust_parse_json
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()

class BaseAgent(ABC):
    """Agent基类 - 定义通用的agent执行模式"""
    
    def __init__(self, 
                 tool_manager: Optional[ToolManager] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        """
        初始化Agent
        
        Args:
            tool_manager: 工具管理器
            model_name: 模型名称
            temperature: 模型温度
            max_tokens: 最大token数
        """
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tool_mode = "auto"  # 默认工具选择模式，可扩展为 "auto", "required", "none"
    
    @property
    @abstractmethod
    def role_name(self) -> str:
        """角色名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def system_prompt_template_name(self) -> str:
        """系统提示词模板名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def task_prompt_template_name(self) -> str:
        """任务提示词模板名称 - 子类必须实现"""
        pass
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取任务提示词参数 - 子类可重写
        
        Args:
            pre_tool_results: 前置工具结果
            
        Returns:
            提示词参数字典
        """
        return pre_tool_results
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """
        解析结果 - 子类可重写自定义解析逻辑
        
        Args:
            content: LLM输出内容
            
        Returns:
            解析后的结果
        """
        try:
            parsed = robust_parse_json(content)
            log.info(f"{self.role_name} 结果解析成功")
            return parsed
        except ValueError as e:
            log.warning(f"JSON解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"解析过程出错: {e}")
            return {"raw": content}
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """获取默认前置工具结果 - 子类可重写"""
        return {}
    
    async def execute_pre_tools(self, state: DFState) -> Dict[str, Any]:
        """执行前置工具"""
        log.info(f"开始执行 {self.role_name} 的前置工具...")
        
        if not self.tool_manager:
            log.info("未提供工具管理器，使用默认值")
            return self.get_default_pre_tool_results()
        
        results = await self.tool_manager.execute_pre_tools(self.role_name)
        
        # 设置默认值
        defaults = self.get_default_pre_tool_results()
        for key, default_value in defaults.items():
            if key not in results or results[key] is None:
                results[key] = default_value
                
        log.info(f"前置工具执行完成，获得: {list(results.keys())}")
        return results
    
    def build_messages(self, 
                      state: DFState, 
                      pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """构建消息列表"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        # log.info(f"系统提示词: {sys_prompt}")
        log.info(f"任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def create_llm(self, state: DFState, bind_post_tools: bool = False) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        if bind_post_tools and self.tool_manager:
            post_tools = self.get_post_tools()
            if post_tools:
                llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
                log.info(f"为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
        return llm
    
    def get_post_tools(self) -> List[Tool]:
        """获取后置工具列表"""
        if not self.tool_manager:
            return []
        return self.tool_manager.get_post_tools(self.role_name)
    
    async def process_simple_mode(self, state: DFState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """简单模式处理"""
        log.info(f"执行 {self.role_name} 简单模式...")
        
        messages = self.build_messages(state, pre_tool_results)
        llm = self.create_llm(state, bind_post_tools=False)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content
            log.info(answer_text)
            log.info("LLM调用成功，开始解析结果")
        except Exception as e:
            log.exception("LLM调用失败: %s", e)
            return {"error": str(e)}
        
        return self.parse_result(answer_text)
    
    async def process_with_llm_for_graph(self, messages: List[BaseMessage], state: DFState) -> BaseMessage:
        # filtered_messages = []
        # for msg in messages:
        #     if not filtered_messages and hasattr(msg, 'type') and msg.type == "tool":
        #         continue  # 跳过开头的 tool 消息
        #     filtered_messages.append(msg)
        
        llm = self.create_llm(state, bind_post_tools=True)
        try:
            response = await llm.ainvoke(messages)  # 使用过滤后的消息
            log.info(response)
            log.info(f"{self.role_name} 图模式LLM调用成功")
            return response
        except Exception as e:
            log.exception(f"{self.role_name} 图模式LLM调用失败: {e}")
            raise
    
    def has_tool_calls(self, message: BaseMessage) -> bool:
        """检查消息是否包含工具调用"""
        return hasattr(message, 'tool_calls') and bool(getattr(message, 'tool_calls', None))
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """
        更新状态结果 - 子类可重写以自定义状态更新逻辑
        
        Args:
            state: 状态对象s
            result: 处理结果
            pre_tool_results: 前置工具结果
        """
        # 默认行为：将结果存储到与角色名对应的属性中
        setattr(state, self.role_name.lower(), result)
        state.agent_results[self.role_name] = {
            "pre_tool_results": pre_tool_results,
            "post_tools": [t.name for t in self.get_post_tools()],
            "results": result
        }
    
    async def execute(self, state: DFState, use_agent: bool = False, **kwargs) -> DFState:
        """
        统一执行入口
        
        Args:
            state: DFState实例
            use_agent: 是否使用代理模式
            **kwargs: 额外参数
            
        Returns:
            更新后的DFState
        """
        log.info(f"开始执行 {self.role_name}")
        
        try:
            # 1. 执行前置工具
            pre_tool_results = await self.execute_pre_tools(state)
            
            # 2. 检查是否有后置工具
            post_tools = self.get_post_tools()
            
            # 3. 根据模式和工具情况选择处理方式
            if use_agent and post_tools:
                log.info("代理模式 - 需要外部图构建器处理，暂存必要数据")
                if not hasattr(state, 'temp_data'):
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
                state.temp_data[f'{self.role_name}_instance'] = self
                log.info(f"已暂存前置工具结果和 {self.role_name} 实例，后置工具: {[t.name for t in post_tools]}")
            else:
                if use_agent and not post_tools:
                    log.info("代理模式无可用后置工具，回退到简单模式")
                result = await self.process_simple_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} 执行完成")
            
        except Exception as e:
            log.exception(f"{self.role_name} 执行失败: {e}")
            error_result = {"error": str(e)}
            self.update_state_result(state, error_result, {})
            
        return state
    
    def create_assistant_node_func(self, state: DFState, pre_tool_results: Dict[str, Any]):
        """创建助手节点函数（供图构建器使用）"""
        async def assistant_node(graph_state):
            messages = graph_state.get("messages", [])
            
            if not messages:
                messages = self.build_messages(state, pre_tool_results)
                log.info(f"构建 {self.role_name} 初始消息，包含前置工具结果")
            
            response = await self.process_with_llm_for_graph(messages, state)
            
            if self.has_tool_calls(response):
                log.info(f"{self.role_name} LLM选择调用工具: {[call.get('name', 'unknown') for call in response.tool_calls]}")
                return {"messages": messages + [response]}
            else:
                log.info(f"{self.role_name} LLM未调用工具，解析最终结果")
                result = self.parse_result(response.content)
                return {
                    "messages": messages + [response],
                    self.role_name.lower(): result,
                    "finished": True
                }
        
        return assistant_node
    # def create_assistant_node_func(
    #     self,
    #     state: DFState,
    #     pre_tool_results: Dict[str, Any],
    #     *,
    #     result_aliases: Optional[List[str]] = None,  # 允许在构建节点时传入别名
    # ):
    #     """
    #     创建助手节点函数（供 LangGraph 使用）
    #     - 默认将结果写入 DFState.<role_name.lower()>
    #     - 可通过 result_aliases 或子类覆写 alias_result_keys 来额外同步写入其它 DFState 字段
    #     """
    #     # 子类可覆写该方法来自定义别名集合
    #     def alias_result_keys(st: DFState) -> List[str]:
    #         # 子类可覆写成: return ["recommendation"] 之类
    #         return []

    #     primary_key = self.role_name.lower()
    #     extra_keys = result_aliases or alias_result_keys(state)

    #     async def assistant_node(graph_state: DFState):
    #         # 1) 读取或构造消息
    #         msgs = getattr(graph_state, "messages", [])
    #         if not msgs:
    #             msgs = self.build_messages(state, pre_tool_results)

    #         # 2) 调用 LLM（可含工具）
    #         response = await self.process_with_llm_for_graph(msgs, state)

    #         # 3) 写回消息
    #         new_msgs = msgs + [response]
    #         try:
    #             graph_state.messages = new_msgs
    #         except Exception:
    #             pass

    #         # 4) 工具调用则交给工具节点
    #         if self.has_tool_calls(response):
    #             return {"messages": new_msgs}

    #         # 5) 解析最终结果（robust_parse_json 内部已处理 ```json 包裹）
    #         result = self.parse_result(response.content)
    #         # ops -> oplist 的兼容处理（可选）
    #         if isinstance(result, dict) and "oplist" not in result and isinstance(result.get("ops"), list):
    #             result["oplist"] = result["ops"]

    #         # 6) 将结果写入 DFState：主键 + 额外别名键（仅对存在的字段生效）
    #         # 6.1 主键：<role_name.lower()>
    #         try:
    #             setattr(graph_state, primary_key, result)
    #         except Exception:
    #             pass
    #         # 6.2 别名键：子类或调用方传入
    #         for k in extra_keys:
    #             if hasattr(graph_state, k):
    #                 try:
    #                     setattr(graph_state, k, result)
    #                 except Exception:
    #                     pass

    #         # 7) 返回合并增量（LangGraph 需要）
    #         out = {"messages": new_msgs}
    #         # 仅返回 DFState 上真实存在的键，避免 TypedState 合并报错
    #         if hasattr(graph_state, primary_key):
    #             out[primary_key] = result
    #         for k in extra_keys:
    #             if hasattr(graph_state, k):
    #                 out[k] = result

    #         # 若你的 DFState 有 finished 字段，可在此置位（同样用 hasattr 防御）
    #         if hasattr(graph_state, "finished"):
    #             try:
    #                 graph_state.finished = True
    #             except Exception:
    #                 pass
    #             out["finished"] = True

    #         return out

    #     return assistant_node