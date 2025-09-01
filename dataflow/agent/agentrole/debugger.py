#!/usr/bin/env python3
"""
debugger.py  ── DebugAgent: LLM-driven debugging, validation, and repair of agent task outputs
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module defines DebugAgent, a tool for validating, debugging, and repairing the output of agent tasks using LLMs.

Features:
* Automated LLM-based code and JSON debugging with template-based correction prompts.
* End-to-end interaction with agent memory and task context.
* Flexible integration into agent pipelines as a drop-in component for output validation and repair.
* Robust JSON structure comparison and key validation against prompt templates.
* Supports iterative correction loops for LLM-generated outputs that do not conform to required schemas.

Designed for agent workflow applications that require reliable and automated output validation and correction.

Thread-safety: DebugAgent instances are not inherently thread-safe and should be used accordingly in concurrent environments.
"""
import json
import requests
from typing import Dict, Any,List

 

from ..taskcenter.task_dispatcher import Task
from json.decoder import JSONDecoder
from json.decoder import JSONDecodeError
from ..servicemanager.memory_service import Memory,MemoryClient
from ..toolkits import (ChatAgentRequest,local_tool_for_clean_json)
from dataflow import get_logger
logger = get_logger()

class DebugAgent:
    def __init__(self, task_chain:List[Task], memory_entity:Memory, request:ChatAgentRequest):
        self.task_chain = task_chain
        self.task = self.task_chain[-1]
        self.headers = {
            "Authorization": f"Bearer {self.task.api_key}",
            "Content-Type": "application/json"
        }
        self.memory = memory_entity
        self.client = MemoryClient(self.memory)
        self.prompt_generator = self.task.prompts_template
        self.json_template_prompts = self.prompt_generator.json_form_templates
        self.request = request
        self.json_template_keys = {}
    async def llm_caller(self,prompts:str):
        json_data = {
            "model": 'o3', #request.modelname
            "messages": [
                {"role": "system", "content": "你是一个python代码debug专家"},
                {"role": "user", "content": prompts}
            ],
        }
        content = await self.client.post(
            url=f"{self.task.base_url}",
            headers=self.headers,
            json_data=json_data,
            session_key= self.request.sessionKEY
        )
        # content = result["choices"][0]["message"]["content"]
        return content

    async def debug_simple_tool_code(self, func_name, code, error, history):
        prompt = self._build_debug_prompt(func_name, code, error, history)
        debug_info = await self.llm_caller(prompt)
        return debug_info

    def _build_debug_prompt(self, func_name, code, error, history):
        prompt = self.prompt_generator.render_code_debug("task_prompt_for_code_debug_fix",
                                              func_name = func_name,
                                              code = code,
                                              error = error,
                                              history = history
                                              )
        return prompt
    
    async def debug_pipeline_tool_code(self, template_name, code, error, cls_detail_code, history, data_keys):
        prompt = self.prompt_generator.render_code_debug(template_name,
                                                         code = code,
                                                         error = error ,
                                                         cls_detail_code = cls_detail_code,
                                                         history = history,
                                                         data_keys = data_keys)
        debug_info = await self.llm_caller(prompt)
        return debug_info

    async def debug_form(self, key, result_json_str):
        # 获取模板
        template_prompt = self.json_template_prompts.get(key)
        if not template_prompt:
            logger.debug(f"-----任务{key}的debug_form,没有对应模板提示词！-----")
            return result_json_str
        # 检查合法性
        try:
            result_obj = json.loads(result_json_str)
        except Exception as e:
            error = f"返回内容不是合法的JSON格式，错误：{str(e)}"
            return await self._llm_repair_json(key, result_json_str, error, template_prompt)
        # 检查字段
        try:
            template_json = self._extract_json_from_prompt(template_prompt)
        except Exception as e:
            raise ValueError(f"模板prompts中无法解析出JSON模板，错误：{str(e)}")

        missing_keys, extra_keys = self._compare_keys(result_obj, template_json)
        if missing_keys or extra_keys:
            error = (
                f"JSON结果与模板不一致：缺失字段{missing_keys}，多余字段{extra_keys}。"
                f"模板为：{json.dumps(template_json, ensure_ascii=False)}"
            )
            return await self._llm_repair_json(key, result_json_str, error, template_prompt)

        # allowed_tpl =  [
        #     # 顶层
        #     "info",
        #     # info → context
        #     "context",
        #     # context → edges / nodes / reason / outputs
        #     "edges", "nodes", "reason", "outputs",
        #     # edges
        #     "source", "target",
        #     # nodes
        #     "name", "description", "id", "type", "command", "required", "depends_on",
        #     # outputs → branch_handling
        #     "branch_handling",
        #     # branch_handling 里的字段
        #     "from", "description", "condition", "action",
        #     "type", "value"
        # ],
        # result_json_str = local_tool_for_clean_json(result_json_str,allowed_tpl,concat_keys={"reason"},hoist_children_spec={"outputs":[]})
        return result_json_str

    def _extract_json_from_prompt(self, prompt):

        import re
        match = re.search(r'(\{.*\})', prompt, re.DOTALL)
        if not match:
            raise ValueError("模板prompts中未找到JSON结构")
        json_str = match.group(1)
        # 只取最外层
        return json.loads(json_str)

    def _compare_keys(self, result_obj, template_obj):

        if not isinstance(result_obj, dict) or not isinstance(template_obj, dict):
            raise ValueError("只支持dict类型模板和结果校验")
        result_keys = set(result_obj.keys())
        template_keys = set(template_obj.keys())
        missing_keys = list(template_keys - result_keys)
        extra_keys = list(result_keys - template_keys)
        return missing_keys, extra_keys

    async def _llm_repair_json(self, key, result_json_str, error, template_prompt):

        # prompt = (
        #     f"You are a strict result validation and correction Agent.\n"
        #     f"This is the returned content for task {key}:\n{result_json_str}\n"
        #     f"The encountered issue is:\n{error}\n"
        #     f"Please refer to the following JSON template and help me automatically fix the returned content so that it strictly conforms to the template (keys, types, no extra or missing fields):\n"
        #     f"{template_prompt}\n"
        #     f"Return a JSON string that strictly conforms to the template."
        # )
        prompt = self.prompt_generator.render_json_form(key,result_json_str = result_json_str)
        fixed_json = await self.llm_caller(prompt)
        # allowed_tpl=[
        #     # 顶层
        #     "info",
        #     # info → context
        #     "context",
        #     # context → edges / nodes / reason / outputs
        #     "edges", "nodes", "reason", "outputs",
        #     # edges
        #     "source", "target",
        #     # nodes
        #     "name", "description", "id", "type", "command", "required", "depends_on",
        #     # outputs → branch_handling
        #     "branch_handling",
        #     # branch_handling 里的字段
        #     "from", "description", "condition", "action",
        #     "type", "value"
        # ]
        #
        # fixed_json = local_tool_for_clean_json(fixed_json,allowed_tpl,concat_keys={"reason"},hoist_children_spec={"outputs":[]})

        logger.debug(f"--------校正结果：--------{fixed_json}")
        return fixed_json