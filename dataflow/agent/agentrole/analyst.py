#!/usr/bin/env python3
"""
analyst.py  ── AnalystAgent: LLM-driven analysis and robust JSON parsing for agent workflows
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-03

This module defines the AnalystAgent, an agent component for generating analysis results using LLMs,
managing conversational context, and robustly extracting structured data from responses.

Features:
* Handles chat-based analysis tasks with flexible session and memory management.
* Integrates optional debug agent for iterative LLM output correction.
* Supports robust JSON parsing with comment and trailing comma removal, enabling extraction and merging of multiple JSON objects from model output.
* Designed for modular use in agent pipelines and dataflow systems.

Thread-safety: AnalystAgent instances are not inherently thread-safe and should be managed accordingly in concurrent environments.
"""
from typing import Dict, Any
from ..taskcenter.task_dispatcher import Task
from json.decoder import JSONDecoder, JSONDecodeError
from ..servicemanager.memory_service import Memory, MemoryClient
from ..toolkits import (  
    ChatAgentRequest
)
import re
from dataflow import get_logger
logger = get_logger()

class AnalystAgent:
    def __init__(
        self,
        task: Task,
        memory_entity: Memory,
        debug_agent=None
    ):
        """
        Initialize the AnalystAgent with task, memory entity, and optional debug agent.
        """
        self.task = task
        self.headers = {
            "Authorization": f"Bearer {self.task.api_key}",
            "Content-Type": "application/json"
        }
        self.memory = memory_entity
        self.client = MemoryClient(self.memory)
        self.debug_agent = debug_agent

    async def generate_analysis_results(
        self,
        request: ChatAgentRequest
    ) -> Dict[str, Any]:
        """
        Generate analysis results using the provided chat request.
        """
        task_description = self.task.task_prompts
        logger.debug(f"[{self.task.task_name}'s task promots]: {task_description}")
        if task_description is None:
            return {}
        logger.debug(f"[Previous task result]:{self.task.pre_task_result}!")
        json_data = {
            "model": self.task.modelname,
            "messages": [
                {"role": "system", "content": self.task.sys_prompts},
                {"role": "user", "content": f"Context information: {self.task.pre_task_result}"
                                            f"\n{task_description}"}
            ]
        }
        session_key = request.sessionKEY
        content = await self.client.post(
            url=f"{self.task.base_url}",
            headers=self.headers,
            json_data=json_data,
            session_key=session_key
        )
        if self.debug_agent:
            logger.info(f"[Task name]: {self.task.task_name}")
            content = await self.debug_agent.debug_form(self.task.task_name, content)
        self.task.task_result = self.robust_parse_json(content)
        logger.info(f"[parse results]:  {self.task.task_result}")
        return self.task.task_result
    
    def _strip_json_comments(self, s: str) -> str:
        """
        Remove block and line comments, and trailing commas from JSON-like strings.
        """
        # /*  ...  */
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
        # // ...   （仅限行首或前面只有空白）
        s = re.sub(r'^\s*//.*$', '', s, flags=re.MULTILINE)
        # 尾逗号  ,}
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    def robust_parse_json(self, s: str) -> dict:
        """
        Robustly parse one or more JSON objects from a string.
        Merges multiple dicts if multiple JSON objects are found.
        """
        clean = self._strip_json_comments(s)
        decoder = JSONDecoder()
        idx = 0
        dicts = []
        length = len(clean)
        while True:
            idx = clean.find('{', idx)
            if idx < 0 or idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(clean, idx)
                if isinstance(obj, dict):
                    dicts.append(obj)
                idx = end
            except JSONDecodeError:
                idx += 1
        if not dicts:
            raise ValueError("No JSON object extracted from the input")
        if len(dicts) == 1:
            return dicts[0]
        merged: Dict[str, Any] = {}
        for d in dicts:
            merged.update(d)
        return merged