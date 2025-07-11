#!/usr/bin/env python3
"""
task_dispatcher.py  ── Task object with dynamic template and function serialization support
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module provides the `Task` class, which encapsulates:

* Dynamic template rendering for system and task prompts
* Serialization/deserialization of non-picklable function references and objects
* Task configuration, parameters, dependencies, and result processing logic
* Integration with YAML config files and prompt template generators

It is designed for flexible agent orchestration and supports advanced
workflows with custom function injection and template management.

Thread-safety depends on user code and injected functions.
"""

import json
import requests
import yaml
from typing import Dict, Any, Iterable
import os
from enum import Enum
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
import importlib
import inspect
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Union, List, Set
FuncSpec = Dict[str, str]
def func2spec(fn: Callable) -> FuncSpec:
    """
    Convert a *module-level* function to a JSON/pickle-friendly spec.
    """
    mod = fn.__module__
    qual = fn.__qualname__
    return {"module": mod, "qualname": qual}

def spec2func(spec: FuncSpec) -> Callable:
    """
    Restore a function object from its spec.
    """
    mod = importlib.import_module(spec["module"])
    fn  = mod
    for attr in spec["qualname"].split("."):
        fn = getattr(fn, attr)
    return fn

def is_module_level_fn(fn: Any) -> bool:
    """
    Check if a function is a module-level function (not a bound method).
    """
    return inspect.isfunction(fn) and fn.__qualname__ == fn.__name__


@dataclass
class TaskChainConfig:
    # ── Router 分流时可保留的任务名 ──
    pipeline_tasks: Set[str] = field(
        default_factory=lambda: {
            "data_content_classification",
            "recommendation_inference_pipeline",
            "execute_the_recommended_pipeline",
        }
    )
    operator_tasks: Set[str] = field(
        default_factory=lambda: {
            "match_operator",
            "write_the_operator",
        }
    )
    # ── 传给 generate_pre_task_params_with_sandboxed_prompt_param_builder 的调试工具 ──
    debuggable_tools: Dict[str, bool] = field(
        default_factory=lambda: {
            "local_tool_for_execute_the_recommended_pipeline": True,
        }
    )

def build_cfg(tasks: Iterable) -> TaskChainConfig:
    pipeline_tasks, operator_tasks, dbg_tools = set(), set(), {}
    for t in tasks:
        match getattr(t, "category", None):
            case "pipeline":  pipeline_tasks.add(t.task_name)
            case "operator":  operator_tasks.add(t.task_name)

        dbg_tools.update(getattr(t, "need_debug_tools", {}))

    return TaskChainConfig(
        pipeline_tasks  = pipeline_tasks,
        operator_tasks  = operator_tasks,
        debuggable_tools= dbg_tools
    )

class Task:
    def __init__(self, request, config_path, prompts_template: PromptsTemplateGenerator,
                 system_template: str, task_template: str,
                 param_funcs: dict, is_result_process = False, task_result_processor = None,
                 use_pre_task_result = True, task_name = None):
        # ------------------- Configuration -------------------
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.config_path      = config_path
        # self.api_key          = config["task_api_key"]
        # self.base_url         = config["task_chat_api_url"]
        # self.modelname        = config["task_model"]
        self.api_key          = request.api_key
        self.base_url         = request.chat_api_url
        self.modelname        = request.model

        task_entry = config.get(task_name, {}) if task_name else {}
        if "special_model" in task_entry:
            self.modelname = task_entry.pop("special_model") or self.modelname

        # self.tasktype      = TaskType(config["tasktype"])
        # ------------------- Runtime fields -------------------
        self.prompts_template   = prompts_template
        self.param_funcs        = param_funcs         # May contain non-picklable objects
        self.task_result_processor = task_result_processor
        # ------------------- Data fields -------------------
        self.task_name          = task_name
        self.system_template    = system_template
        self.task_template      = task_template
        self.sys_prompts        = ""
        self.task_prompts       = ""
        self.done               = False
        self.task_result        = ""
        self.pre_task_result: Union[Dict[str, Any], List[Dict[str, Any]]] = {}
        self.use_pre_task_result = use_pre_task_result
        self.depends_on         = [] # Task dependencies, default is previous task result
        self.task_params        = None
        self.is_result_process  = is_result_process
        self.category           = ''
        self.need_debug_tools   = {}
        if self.task_name in config:
            self.tool_call_map      = config[self.task_name].get('tools', {})
            self.category           = config[self.task_name].get('category', '')
            self.need_debug_tools   = config[self.task_name].get('need_debug_tools',{})
        else:
            self.tool_call_map  = {}
        
    def render_templates(self):
        """
        Dynamically render all templates based on task parameters.
        """
        if not self.task_params or "templates" not in self.task_params:
            print("No templates to render.")
            return
        for template_cfg in self.task_params["templates"]:
            template_name = template_cfg["name"]
            params = template_cfg.get("params", {})
            try:
                rendered = self.prompts_template.render(
                    template_name=template_name,
                    **params
                )
                if "system" in template_name.lower():
                    self.sys_prompts = rendered
                else:
                    self.task_prompts = rendered
            except Exception as e:
                print(f"Error rendering template {template_name}: {str(e)}")

    def __getstate__(self):
        """
        Make the object serializable for pickling. Save function references as specs.
        """
        state = self.__dict__.copy()
        state["_prompts_template_cls"] = (
            self.prompts_template.__class__.__module__,
            self.prompts_template.__class__.__qualname__,
        )
        state["prompts_template"] = None
        serializable_funcs = {}
        for name, fn in self.param_funcs.items():
            if is_module_level_fn(fn):
                serializable_funcs[name] = func2spec(fn)
            else:
                # For bound-methods like local_tool_for_sample, save a placeholder
                serializable_funcs[name] = {"bound_sampler": True}
        state["param_funcs"] = serializable_funcs
        trp = self.task_result_processor
        if trp and is_module_level_fn(trp):
            state["task_result_processor"] = func2spec(trp)
            state["_trp_is_spec"] = True
        else:
            state["_trp_is_spec"] = False
        return state

    def __setstate__(self, state):
        """
        Restore object state from a pickle, including functions and template generator.
        """
        self.__dict__.update(state)
        # ---- prompts_template ----
        mod_name, qual_name = self.__dict__.pop("_prompts_template_cls")
        mod = importlib.import_module(mod_name)
        cls = mod
        for attr in qual_name.split("."):
            cls = getattr(cls, attr)
        self.prompts_template = cls()       # Re-instantiate template generator
        # ---- param_funcs ----
        restored = {}
        for name, spec in self.param_funcs.items():
            if spec == {"bound_sampler": True}:
                # Requires a new Sampler instance
                from ..toolkits import local_tool_for_sample
                restored[name] = local_tool_for_sample
            else:
                restored[name] = spec2func(spec)
        self.param_funcs = restored

        # ---- task_result_processor ----
        if self.__dict__.pop("_trp_is_spec"):
            self.task_result_processor = spec2func(self.task_result_processor)