#!/usr/bin/env python3
"""
task_registry.py  ── Registry and factory for Task instances in agent pipelines
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module provides a TaskRegistry class for managing and instantiating Task objects by name.

Features:
* Decorator-based registration of task factories.
* Retrieval of Task instances by name, with support for custom prompt templates.
* Listing of all registered tasks.
* Ensures unique task names and clear error handling for unregistered tasks.

Intended for extensible agent pipelines where tasks are defined, registered, and instantiated dynamically.

Thread-safety: Registry modification is not thread-safe by default and should be synchronized if used in concurrent environments.
"""
from __future__ import annotations
import inspect
from typing import Callable, Dict, Any

from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from .task_dispatcher import Task

class TaskRegistry:
    """
    注册/获取 Task 工厂的统一入口，支持根据工厂函数签名自动注入额外上下文
    （如 ChatAgentRequest、Memory、Logger 等）。
    """

    # name  -> factory
    _factories: Dict[str, Callable[..., Task]] = {}

    # ---------- 注册 ---------- #
    @classmethod
    def register(cls, name: str):
        """
        用法:
        @TaskRegistry.register('my_task')
        def _make_my_task(prompts_template, request: ChatAgentRequest = None):
            ...
        """
        def decorator(factory: Callable[..., Task]):
            if name in cls._factories:
                raise KeyError(f"Task '{name}' is already registered")
            cls._factories[name] = factory
            return factory
        return decorator

    @classmethod
    def get(
        cls,
        name: str,
        prompts_template: PromptsTemplateGenerator,
        **context: Any,                    # 额外的上下文(如 request=request)
    ) -> Task:
        """
        根据工厂函数的形参决定是否注入对应的上下文对象。
        1. 始终注入 prompts_template
        2. 如果形参里出现了与 context key 同名的参数，则自动传入
        """
        factory = cls._factories.get(name)
        if factory is None:
            raise KeyError(
                f"No task named '{name}' was found. "
                "Please register it first."
            )

        sig = inspect.signature(factory)
        kwargs: dict[str, Any] = {}

        # 总是注入 prompts_template，只在工厂需要时才包含
        if "prompts_template" in sig.parameters:
            kwargs["prompts_template"] = prompts_template

        # 其余上下文按需注入
        for k, v in context.items():
            if k in sig.parameters:
                kwargs[k] = v

        return factory(**kwargs)

    # ---------- 工具 ---------- #
    @classmethod
    def list_tasks(cls):
        return list(cls._factories.keys())