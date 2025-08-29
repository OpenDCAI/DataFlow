import importlib
import os
import uuid
from json import JSONDecoder, JSONDecodeError
from ..taskcenter import Task
import yaml
import json
import requests
from typing import Dict, Any
from ..servicemanager.memory_service import Memory, MemoryClient
from ..toolkits import (
    ChatResponse,
    ChatAgentRequest,
    ToolRegistry,
)

from ..promptstemplates import PromptsTemplateGenerator
import re
from dataflow import get_logger
logger = get_logger()
class PlanningAgent:
    def __init__(
        self,
        request: ChatAgentRequest,
        config_path: str,
        memory_entity: Memory,
        prompt_template: PromptsTemplateGenerator,
        toolkit: ToolRegistry,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.api_key = cfg["API_KEY"]
        self.api_url = cfg["CHAT_API_URL"]
        self.model = cfg["MODEL"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.prompt_template = prompt_template
        self.memory_entity = memory_entity
        self.client = MemoryClient(self.memory_entity)
        self.request = request
        self.predefined_tools = toolkit
        self.generated_tools = None

        self.PARAM_FUNC_MAP = {name: tool.func for name, tool in toolkit.tools.items()}
        self.PROCESSOR_MAP = {
            # "combine_pipeline_result": combine_pipeline_result,
            # "generate_python_plot": your_generate_python_plot_func,
        }

    async def plan(self, query: str) -> Dict[str, Any]:
        system_prompt = self.prompt_template.templates["system_prompt_for_planer"]
        user_prompt = self.prompt_template.render(
            "task_prompt_for_planer",
            query=query,
            tools_info=self.predefined_tools.tools_info_as_json(),
        )

        json_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # 使用 MemoryClient 异步 post
        content = await self.client.post(
            url=self.api_url,
            headers=self.headers,
            json_data=json_data,
            session_key=self.request.sessionKEY,
        )
        return self.robust_parse_json(content)

    def build_task_chain(self, json_data, config_path):
        task_chain = []
        for task_cfg in json_data["tasks"]:
            # param_funcs: ["star_count", "fork_count"] -> {"star_count": func, ...}
            param_funcs = {}
            for param in task_cfg.get("param_funcs", []):
                if param in self.PARAM_FUNC_MAP and self.PARAM_FUNC_MAP[param] is not None:
                    param_funcs[param] = self.PARAM_FUNC_MAP[param]
                else:
                    # 也可以直接传字符串或 lambda x: x
                    param_funcs[param] = f"{param}"

            logger.debug(f"Task config: {task_cfg}")
            logger.debug(f"Task template: {task_cfg['task_template']}")

            processor = None
            if "task_result_processor" in task_cfg:
                processor_name = task_cfg["task_result_processor"]
                processor = self.PROCESSOR_MAP.get(processor_name)

            task = Task(
                config_path=config_path,
                prompts_template=self.prompt_template,
                system_template=task_cfg["system_template"],
                task_template=task_cfg["task_template"],
                param_funcs=param_funcs,
                is_result_process=task_cfg.get("is_result_process", False),
                task_result_processor=processor,
                use_pre_task_result=task_cfg.get("use_pre_task_result", False),
                task_name=task_cfg["name"],
            )
            task.depends_on = task_cfg.get("depends_on", [])
            task_chain.append(task)

        for prompt_dict in json_data.get("prompts", []):
            for key, value in prompt_dict.items():
                self.prompt_template.templates[key] = value
        return task_chain

    def _strip_json_comments(self, s: str) -> str:
        # # 1) 去掉多行注释 /* … */
        # s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
        # # 2) 去掉单行注释 //…
        # s = re.sub(r'//.*$', '', s, flags=re.MULTILINE)
        # # 3) 去掉尾随逗号（最后一个元素后面多余的逗号）
        # #        { "a":1, } -> { "a":1 }
        # s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    def robust_parse_json(self, s: str) -> dict:
        clean = self._strip_json_comments(s)
        decoder = JSONDecoder()
        idx = 0
        dicts = []
        length = len(clean)
        while True:
            idx = clean.find("{", idx)
            if idx < 0 or idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(clean, idx)
                if isinstance(obj, dict):
                    dicts.append(obj)
                idx = end
            except JSONDecodeError:
                # 如果当前位置无法解析，跳过去继续找下一个 '{'
                idx += 1
        if not dicts:
            raise ValueError("在输入中未提取到任何 JSON 对象")
        if len(dicts) == 1:
            return dicts[0]
        # 多个 JSON 时合并到一个 dict（后面的同键值覆盖前面）
        merged: Dict[str, Any] = {}
        for d in dicts:
            merged.update(d)
        return merged


if __name__ == "__main__":
    pass