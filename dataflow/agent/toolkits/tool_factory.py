#!/usr/bin/env python3
"""
tool_factory.py ── Tool & ToolRegistry: A simple registry for reusable callable tools
Author  : [Your Name]
License : MIT
Created : 2024-07-10

This module defines the Tool and ToolRegistry classes, providing a lightweight framework for registering, managing, and invoking reusable callable tools within a Python application.

Features:
* Simple decorator-based tool registration for functions.
* Centralized registry for tool lookup and invocation by name.
* Optional tool typing and descriptive documentation support.
* Easy JSON export of tool metadata for inspection or UI integration.
* Supports extension to remote or advanced tool types with minimal changes.

Designed for agent, pipeline, or utility scenarios that require modular, discoverable, and well-documented function management.

Thread-safety: ToolRegistry is not inherently thread-safe. For concurrent environments, use appropriate synchronization mechanisms.
"""
from typing import Callable, Any, Dict
from functools import wraps

class Tool:
    def __init__(self, name: str, func: Callable, type_: str = "local", desc: str = ""):
        self.name = name
        self.func = func
        self.type = type_
        self.desc = desc

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, func: Callable, type_: str = "local", desc: str = ""):
        if name in self.tools:
            return
        self.tools[name] = Tool(name, func, type_, desc)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found.")
        return self.tools[name]

    def call(self, name: str, *args, **kwargs):
        tool = self.get(name)
        return tool(*args, **kwargs)

    def list_tools(self, type_: str = None):
        if type_:
            return [tool for tool in self.tools.values() if tool.type == type_]
        return list(self.tools.values())

    def tools_info_as_json(self) -> str:
        import json
        tools_info = [
            {"name": tool.name, "desc": tool.desc}
            for tool in self.tools.values()
        ]
        return json.dumps(tools_info, ensure_ascii=False, indent=2)
tool_registry = ToolRegistry()
def TOOL(desc: str = "", type_: str = "local"):
    def decorator(func):
        tool_registry.register(func.__name__, func, type_=type_, desc=desc)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator