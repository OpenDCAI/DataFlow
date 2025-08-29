"""
analysis_service.py  ── Analysis service for multi-step conversation & task chain
Author  : [Zhou Liu]
License : MIT
Created : 2024-06-25

This module provides the `AnalysisService` class, a high-level orchestrator for
multi-step, multi-agent conversational analysis chains.

Features:
* Loads configuration and tasks from YAML and Python classes
* Handles chat history, multi-turn memory, and session state using Memory
* Supports flexible prompt template integration and debugging
* Robust JSON response parsing (including noisy model outputs)
* Chains multiple analysis/agent tasks, supporting router/decision flows
* Integrates with FastAPI for HTTP exceptions and responses

It is completely thread-safe in CPython **only if** each request handler
works on its own event-loop thread; otherwise add locking by yourself.
"""
import json
import re
from typing import Dict, Any
from json.decoder import JSONDecoder
from json.decoder import JSONDecodeError
from fastapi import HTTPException, status
import httpx
import uuid
# generate_pre_task_params
from ..toolkits import generate_pre_task_params,ChatResponse,ChatAgentRequest,generate_pre_task_params_with_sandboxed_prompt_param_builder
from ..servicemanager.memory_service import Memory
from ..agentrole.analyst import AnalystAgent
from ..agentrole.executioner import ExecutionAgent
from ..agentrole.debugger import DebugAgent
from ..taskcenter import TaskChainConfig
import yaml
import os
 
from dataflow import get_logger
logger = get_logger()

class Config:
    def __init__(self, request:ChatAgentRequest):
        self.API_KEY = request.api_key
        self.CHAT_API_URL = request.chat_api_url
        self.MODEL = request.model
        self.HEADER = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }
class AnalysisService:
    def __init__(self, tasks: list, 
                 memory_entity: Memory, 
                 request: ChatAgentRequest,
                 execution_agent:ExecutionAgent,
                 cfg: TaskChainConfig):
        self.config = Config(request)
        self.memory = memory_entity
        self.request = request
        session_id = self.memory.get_session_id(self.request.sessionKEY)
        restored = self.memory.load_object(session_id, 'service_state', default={})
        self.task_results = restored.get("task_results", [])
        if restored.get("tasks"):
            self.tasks = restored["tasks"]
        else:
            self.tasks = tasks
        self.prompt_template_generator = self.tasks[-1].prompts_template
        self.debug_agent = execution_agent.debug_agent
        self.execution_agent = execution_agent
        self.cfg =  cfg
        self.post_porcess_args = {
            'request': self.request
        }

    async def process_request(self):

        session_id = self.memory.get_session_id(self.request.sessionKEY)
        chat_history = self.memory.get_history(session_id)
        task_results_history = self.memory.get_session_data(session_id,"task_results")
        # logger.debug(f"[chat_history]:{chat_history}")
        # logger.debug(f"[task_results_history]:{task_results_history}")

        if chat_history and self.tasks[0].task_name != "conversation_router":
            logger.info("---------Has History！---------")
            return await self._process_with_history(self.request, session_id, chat_history)
        else:
            return await self._process_with_task_chain(self.request, session_id)

    async def _process_with_history(self, request, session_id, history):
        # Add the user’s latest input to the session memory
        self.memory.add_messages(
            session_id,
            [{"role": "user", "content": request.target}]
        )
        payload  = self._build_history_payload(request, history)
        content  = await self._call_llm_api(payload)
        logger.debug(f"Content before JSON parsing: {content}")
        context_json = self._safe_parse_json(content)
        if self.debug_agent:
            logger.debug(f"context_json before processing: {context_json}")
            context_json = await self.debug_agent.debug_form(
                self.tasks[-1].task_name,
                context_json
            )
            logger.debug(f"context_json after processing: {context_json}")
            context_json = self._safe_parse_json(context_json)
        context_json = self._post_process_result(context_json)
        self._update_memory(session_id, context_json)
        # Save the latest service state in memory
        self.memory.save_object(
            session_id,
            "service_state",
            {
                "task_results": self.task_results,
                "tasks": self.tasks,
            }
        )
        return self._build_response(context_json, content, "ChatAgent")

    async def _process_with_task_chain(self, request, session_id):
        last_result = None
        idx = 0
        while idx < len(self.tasks):
            task = self.tasks[idx]
            task.pre_task_result = last_result  
            # task.task_params = generate_pre_task_params(
            #     system_template=task.system_template,
            #     task_template=task.task_template,
            #     request=request,
            #     param_funcs=task.param_funcs,
            #     pre_task_result=task.pre_task_result if task.use_pre_task_result else None,
            #     memory=self.memory,
            #     extral_param = task.tool_call_map
            # )
            task.task_params =await generate_pre_task_params_with_sandboxed_prompt_param_builder(
                system_template=task.system_template,
                task_template=task.task_template,
                request=request,
                param_funcs=task.param_funcs,
                pre_task_result=task.pre_task_result if task.use_pre_task_result else None,
                memory=self.memory,
                extral_param = task.tool_call_map,
                execution_agent=self.execution_agent,
                debuggable_tools=self.cfg.debuggable_tools,
            )
            task.render_templates()
            agent = AnalystAgent(task, self.memory, self.debug_agent)
            formatted_context = await agent.generate_analysis_results(request)
            last_result = formatted_context
            self.task_results.append(last_result)
            self.post_porcess_args['task_results'] = self.task_results
            self.post_porcess_args['last_result'] = last_result
            if task.task_name == "conversation_router":
                need_rec           = formatted_context.get("need_recommendation",  False)
                need_write_operator = formatted_context.get("need_write_operator", False)
                reply              = formatted_context.get("assistant_reply",     "")

                self.memory.add_messages(session_id, [
                    {"role": "user",      "content": request.target},
                    {"role": "assistant", "content": formatted_context}
                ])
                self.memory.set_session_data(session_id, task.task_name, last_result)

                if (not need_rec) and (not need_write_operator):
                    self.memory.set_session_data(session_id, "last_result", formatted_context)
                    return self._build_response(
                        context  = formatted_context,
                        response = reply or "ok",
                        agent_name = "ChatAgent"
                    )
                if need_rec:
                    valid_tasks = self.cfg.pipeline_tasks
                if need_write_operator and not need_rec:
                    valid_tasks = self.cfg.operator_tasks
                self.tasks = [t for t in self.tasks if t.task_name in valid_tasks]
                self.memory.set_session_data(session_id, "intent_router", formatted_context)
                self.memory.set_session_data(session_id, "_actual_tasks",[t.task_name for t in self.tasks])
                idx = 0 
                continue
            # ---------- end router ----------
            if task.is_result_process:
                last_result = task.task_result_processor(**self.post_porcess_args)
            idx += 1
            logger.info(f"[The final execution result (the result passed to the next task)]:{last_result}")
            self.memory.set_session_data(session_id, task.task_name, last_result)
            logger.info(f'mem.get:{self.memory.get_session_data(session_id, task.task_name)}')
        self.memory.add_messages(session_id, [{"role": "user", "content": request.target}])
        # self.memory.add_response(session_id, {"role": "assistant", "content": last_result})
        self.memory.set_session_data(session_id, 'last_result', last_result)
        # self.memory.set_session_data(session_id, 'task_results', self.task_results)
        payload = self._build_final_chain_payload(request, last_result)
        info = await self._call_llm_api(payload)
        self.memory.add_response(session_id, {"role": "assistant", "content": info})

        if self.tasks[0].task_name == "conversation_router":
            self.tasks = self.tasks[1:]
        self.memory.save_object(session_id, 'service_state', {
            "task_results": self.task_results,
            "tasks": self.tasks,
        })
        return self._build_response(last_result, info, "Analysis Agent")


    def _build_history_payload(self, request, history):
        return {
            "model": request.model,
            "messages": [
                {"role":"system", "content": (f"{self.prompt_template_generator.templates['system_prompt']}")},
                {"role":"user", "content": (
                    f"History: {history}. \n"
                    f"Question: {request.target}.\n"
                    f"(Answer in {request.language})"
                )}
            ],
            **({"temperature": request.temperature, "max_tokens": request.max_tokens} if "gpt" in request.model.lower() else {})
        }

    def _build_final_chain_payload(self, request, last_result):
        return {
            "model": request.model,
            "messages": [
                {"role":"system","content":"Data Analysis Expert"},
                {"role":"user","content": (
                    f"Context: {last_result}\n"
                    f"Question: {request.target}.\n"
                    f"(Answer in {request.language})"
                )}
            ],
            **({"temperature": request.temperature, "max_tokens": request.max_tokens} if "gpt" in request.model.lower() else {})
        }

    async def _call_llm_api(self, payload):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url     = self.config.CHAT_API_URL,
                headers = self.config.HEADER,
                json    = payload,
                timeout = 120.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _safe_parse_json(self, content):
        try:
            return self.robust_parse_json(content)
        except ValueError:
            return {}
    def _post_process_result(self, context_json):
        if self.tasks and getattr(self.tasks[-1], "is_result_process", False):
            try:
                return self.tasks[-1].task_result_processor(context_json,self.task_results)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Task result processor error: {e}"
                )
        return context_json

    def _update_memory(self, session_id, context_json):
        self.memory.add_response(session_id, {"role": "assistant", "content": context_json})
        self.memory.set_session_data(session_id, 'last_result', context_json)
        task_results = self.memory.get_session_data(session_id, 'task_results', [])
        self.memory.set_session_data(session_id, 'task_results', task_results + [context_json])

    def _build_response(self, context, response, agent_name):
        return ChatResponse(
            id=str(uuid.uuid4()),
            name=agent_name,
            info={"context": context, "response": response}
        )
    def _strip_json_comments(self, s: str) -> str:
        # 1) 去掉多行注释 /* … */
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
        # 2) 去掉单行注释 //…
        s = re.sub(r'//.*$', '', s, flags=re.MULTILINE)
        # 3) 去掉尾随逗号（最后一个元素后面多余的逗号）
        #        { "a":1, } -> { "a":1 }
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    def robust_parse_json(self, s: str) -> dict:
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

