# graghbuilder/gragh_builder.py
from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Tuple, Any
from pydantic import BaseModel
from langgraph.graph import StateGraph


class GenericGraphBuilder:
    """
    增强版通用建图器，支持：
    1. @pre_tool 和 @post_tool 装饰器
    2. 链式调用添加节点、边、条件边
    3. 自动工具注册和管理
    """

    def __init__(self, state_model: type[BaseModel], entry_point: str = "start"):
        self.state_model = state_model
        self.entry_point = entry_point
        self.nodes: Dict[str, Tuple[Callable, str]] = {}  # name -> (func, role)
        self.edges: List[Tuple[str, str]] = []
        self.conditional_edges: Dict[str, Callable] = {}
        
        # 工具注册表
        self.pre_tool_registry: Dict[str, Dict[str, Callable]] = {}  # role -> {name: func}
        self.post_tool_registry: Dict[str, List[Callable]] = {}     # role -> [func]
        
        # 延迟导入 tool_manager 避免循环导入
        self.tool_manager = None

    def _get_tool_manager(self):
        """延迟导入 tool_manager"""
        if self.tool_manager is None:
            from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
            self.tool_manager = get_tool_manager()
        return self.tool_manager

    def pre_tool(self, name: str, role: str):
        """装饰器：注册前置工具到指定角色"""
        def decorator(func: Callable):
            if role not in self.pre_tool_registry:
                self.pre_tool_registry[role] = {}
            self.pre_tool_registry[role][name] = func
            return func
        return decorator

    def post_tool(self, role: str):
        """装饰器：注册后置工具到指定角色"""
        def decorator(func: Callable):
            if role not in self.post_tool_registry:
                self.post_tool_registry[role] = []
            self.post_tool_registry[role].append(func)
            return func
        return decorator

    def add_node(self, name: str, func: Callable, role: str = None) -> 'GenericGraphBuilder':
        """添加单个节点，支持链式调用"""
        self.nodes[name] = (func, role or name)
        return self

    def add_nodes(self, nodes: Dict[str, Callable], role_mapping: Dict[str, str] = None) -> 'GenericGraphBuilder':
        """批量添加节点，支持角色映射"""
        role_mapping = role_mapping or {}
        for name, func in nodes.items():
            role = role_mapping.get(name, name)
            self.add_node(name, func, role)
        return self

    def add_edge(self, src: str, dst: str) -> 'GenericGraphBuilder':
        """添加单条边"""
        self.edges.append((src, dst))
        return self

    def add_edges(self, edges: List[Tuple[str, str]]) -> 'GenericGraphBuilder':
        """批量添加边"""
        self.edges.extend(edges)
        return self

    def add_conditional_edge(self, src: str, condition_func: Callable) -> 'GenericGraphBuilder':
        """添加单个条件边"""
        self.conditional_edges[src] = condition_func
        return self

    def add_conditional_edges(self, conditional_edges: Dict[str, Callable]) -> 'GenericGraphBuilder':
        """批量添加条件边"""
        self.conditional_edges.update(conditional_edges)
        return self

    def _register_tools_for_role(self, role: str, state: Any):
        """为指定角色注册工具"""
        tm = self._get_tool_manager()
        
        # 注册前置工具
        if role in self.pre_tool_registry:
            for tool_name, tool_func in self.pre_tool_registry[role].items():
                try:
                    tm.register_pre_tool(
                        name=tool_name,
                        role=role,
                        func=lambda s=state, f=tool_func: f(s),
                        override=True
                    )
                except TypeError:
                    # 兼容不支持 override 参数的版本
                    tm.register_pre_tool(
                        name=tool_name,
                        role=role,
                        func=lambda s=state, f=tool_func: f(s)
                    )

        # 注册后置工具
        if role in self.post_tool_registry:
            for tool_func in self.post_tool_registry[role]:
                tm.register_post_tool(tool_func, role=role)

    def _wrap_node_with_tools(self, node_func: Callable, role: str):
        """为节点包装自动工具注册逻辑"""
        async def wrapped_node(state):
            # 执行前自动注册该角色的工具
            self._register_tools_for_role(role, state)
            
            # 执行原始节点函数
            if asyncio.iscoroutinefunction(node_func):
                return await node_func(state)
            else:
                return node_func(state)
        
        return wrapped_node

    def build(self):
        """构建并返回编译后的图"""
        sg = StateGraph(self.state_model)
        
        # 添加节点（自动包装工具注册逻辑）
        for name, (func, role) in self.nodes.items():
            wrapped_func = self._wrap_node_with_tools(func, role)
            sg.add_node(name, wrapped_func)
        
        # 添加普通边
        for src, dst in self.edges:
            sg.add_edge(src, dst)
        
        # 添加条件边
        for src, cond_func in self.conditional_edges.items():
            sg.add_conditional_edges(src, cond_func)
        
        sg.set_entry_point(self.entry_point)
        return sg.compile()