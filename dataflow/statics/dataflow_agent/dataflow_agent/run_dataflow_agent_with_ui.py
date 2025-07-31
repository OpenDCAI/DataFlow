import os, asyncio
import time
import uuid
from fastapi import FastAPI
import json 
from pathlib import Path
from dataflow.agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.agent.taskcenter import TaskRegistry,TaskChainConfig,build_cfg
import dataflow.agent.taskcenter.task_definitions
from dataflow.agent.servicemanager import AnalysisService, Memory
from dataflow.agent.toolkits import (
    ChatResponse,
    ChatAgentRequest,
    ToolRegistry,
)
from dataflow.agent.agentrole.debugger import DebugAgent
from dataflow.agent.agentrole.executioner import ExecutionAgent
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow import get_logger
from typing import AsyncGenerator, List
from fastapi.responses import StreamingResponse
logger = get_logger()     
toolkit = ToolRegistry()
memorys = {
    "planner": Memory(),
    "analyst": Memory(),
    "executioner": Memory(),
    "debugger": Memory(),
}
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
api_key = os.environ.get("DF_API_KEY", "")
chat_api_url = os.environ.get("DF_API_URL", "")

def _build_task_chain(req: ChatAgentRequest, tmpl:PromptsTemplateGenerator):
    router   = TaskRegistry.get("conversation_router", prompts_template=tmpl,request=req)
    classify = TaskRegistry.get("data_content_classification", prompts_template=tmpl,request=req)
    rec      = TaskRegistry.get("recommendation_inference_pipeline", prompts_template=tmpl,request=req)
    exe      = TaskRegistry.get("execute_the_recommended_pipeline", prompts_template=tmpl,request=req)

    op_match = TaskRegistry.get("match_operator",prompts_template=tmpl, request = req)
    op_write = TaskRegistry.get("write_the_operator",prompts_template=tmpl, request = req)
    op_debug = TaskRegistry.get("exe_and_debug_operator",prompts_template=tmpl, request = req)
    task_chain = [router, classify, rec, exe , op_match, op_write, op_debug]
    cfg      = build_cfg(task_chain)
    return task_chain, cfg

async def _run_service(req: ChatAgentRequest) -> ChatResponse:
    tmpl = PromptsTemplateGenerator(req.language)
    task_chain, chain_cfg = _build_task_chain(req,tmpl = tmpl)
    execution_agent = ExecutionAgent(
                              request           = req,
                              memory_entity     = memorys["executioner"],
                              prompt_template   = tmpl,
                              debug_agent       = DebugAgent(task_chain,memorys["debugger"],req),
                              task_chain        = task_chain
                              )
    
    service = AnalysisService(
        tasks           = task_chain,
        memory_entity   = memorys["analyst"],
        request         = req,
        execution_agent = execution_agent,
        cfg             = chain_cfg
    )
    return await service.process_request()

app = FastAPI(title="Dataflow Agent Service")
# @app.post("/chatagent", response_model=ChatResponse)
# async def chatagent(req: ChatAgentRequest):
#     return await _run_service(req)
@app.post("/chatagent", response_model=ChatResponse)
async def chatagent(req: ChatAgentRequest):
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    if req.chat_api_url:
        os.environ["DF_API_URL"] = req.chat_api_url
    service, _ = _create_service(req)
    return await service.process_request()

def _create_service(req: ChatAgentRequest):
    tmpl                    = PromptsTemplateGenerator(req.language)
    task_chain, chain_cfg   = _build_task_chain(req, tmpl)

    exe_agent = ExecutionAgent(
        request         = req,
        memory_entity   = memorys["executioner"],
        prompt_template = tmpl,
        debug_agent     = DebugAgent(task_chain, memorys["debugger"], req),
        task_chain      = task_chain,
    )

    service = AnalysisService(
        tasks           = task_chain,
        memory_entity   = memorys["analyst"],
        request         = req,
        execution_agent = exe_agent,
        cfg             = chain_cfg,
    )
    return service, task_chain

# @app.post("/chatagent/stream", response_class=StreamingResponse)
# async def chatagent_stream(req: ChatAgentRequest):
#     if req.api_key:
#         os.environ["DF_API_KEY"] = req.api_key
#     if req.chat_api_url:
#         os.environ["DF_API_URL"] = req.chat_api_url
#     service, task_chain = _create_service(req)
#     asyncio.create_task(service.process_request(), name=f"svc-{uuid.uuid4()}")

#     mem: Memory = memorys["analyst"]
#     session_id = mem.get_session_id(req.sessionKEY)

#     async def event_gen() -> AsyncGenerator[bytes, None]:
#         try:
#             task_names: List[str] = [t.task_name for t in task_chain]
#             while True:
#                 actual = mem.get_session_data(session_id, "_actual_tasks")
#                 if actual:
#                     task_names = (["conversation_router"] + [t for t in actual if t != "conversation_router"])
#                     break
#                 await asyncio.sleep(0.1)

#             print('_actual_tasks 有了！！')
#             # ① 连接建立事件
#             yield b'data: {"event":"connected","message":"Stream connected"}\n\n'

#             # ② start 事件
#             for name in task_names:
#                 yield f'data: {json.dumps({"event": "start", "task": name})}\n\n'.encode()

#             # ③ 依次等待每个 task 完成 + 心跳
#             last_ping = time.perf_counter()
#             for name in task_names:
#                 start_ts = time.perf_counter()
#                 while True:
#                     res = mem.get_session_data(session_id, name)
#                     if res is not None:
#                         elapsed = round(time.perf_counter() - start_ts, 3)
#                         evt = {
#                             "event": "finish",
#                             "task": name,
#                             "result": res,
#                             "elapsed": elapsed,
#                         }
#                         yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode()
#                         break
                    
#                     # 添加心跳：每2秒发送一次，防止用户觉得卡住
#                     if time.perf_counter() - last_ping > 2:
#                         yield b'data: {"event":"ping","message":"Processing..."}\n\n'
#                         last_ping = time.perf_counter()
                    
#                     await asyncio.sleep(0.1)  # 缩短间隔到0.1s，提高实时性

#             # ④ done
#             yield b'data: {"event":"done"}\n\n'
#             mem.clear_session(session_id)
#             return

#         except asyncio.CancelledError:
#             return
#         except Exception as e:
#             err = {"event": "error", "detail": repr(e)}
#             yield f"data: {json.dumps(err)}\n\n".encode()
#             yield b'data: {"event":"done"}\n\n'
#             return

#     return StreamingResponse(
#         event_gen(),
#         media_type="text/event-stream",
#         headers={"Cache-Control": "no-cache"},
#     )

@app.post("/chatagent/stream", response_class=StreamingResponse)
async def chatagent_stream(req: ChatAgentRequest):
    # ---------- 环境变量 ----------
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    if req.chat_api_url:
        os.environ["DF_API_URL"] = req.chat_api_url

    # ---------- 启动业务 ----------
    service, task_chain = _create_service(req)
    asyncio.create_task(service.process_request(), name=f"svc-{uuid.uuid4()}")

    mem        = memorys["analyst"]
    session_id = mem.get_session_id(req.sessionKEY)

    # ---------- 初始状态 ----------
    task_names: list[str] = ["conversation_router"]  # 先监听 router
    dynamic_added   = False                           # 是否已追加 _actual_tasks
    router_finished = False                           # router 是否结束
    router_finish_ts  = 0.0
    ROUTER_TIMEOUT    = 30.0                           # router 超时秒数

    # 只想每 INTERVAL_ACTUAL 秒检查一次 _actual_tasks
    INTERVAL_ACTUAL   = 5.0
    last_actual_check = 0.0

    async def event_gen() -> AsyncGenerator[bytes, None]:
        nonlocal task_names, dynamic_added, router_finished, router_finish_ts, last_actual_check

        try:
            sent: set[str] = set()
            start_ts  = time.perf_counter()
            last_ping = start_ts

            # 告诉前端已连接
            yield b'data: {"event":"connected","message":"stream opened"}\n\n'

            # ================ 主循环 ================
            while True:
                now = time.perf_counter()

                # ① 每 INTERVAL_ACTUAL 秒才检查一次 _actual_tasks
                if (not dynamic_added) and (now - last_actual_check >= INTERVAL_ACTUAL):
                    actual = mem.get_session_data(session_id, "_actual_tasks")
                    # print(f'actual : {actual}')
                    last_actual_check = now
                    if actual is not None:
                        task_names = ["conversation_router"] + [
                            t for t in actual if t != "conversation_router"
                        ]
                        dynamic_added = True

                # ② 轮询各任务结果
                all_done = True
                for name in task_names:
                    res = mem.get_session_data(session_id, name)
                    # print(f"name: {name}, Op result {res}")
                    if res is None:
                        all_done = False
                        continue

                    if name not in sent:
                        if name == "conversation_router":
                            router_finished  = True
                            router_finish_ts = now

                        evt = {
                            "event":  "finish",
                            "task":   name,
                            "result": res,
                            "elapsed": round(now - start_ts, 3),
                        }
                        yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode()
                        sent.add(name)

                # ③ router 超时 ⇒ 直接 done
                if (router_finished and not dynamic_added and
                    now - router_finish_ts > ROUTER_TIMEOUT):
                    yield b'data: {"event":"done"}\n\n'
                    mem.clear_session(session_id)
                    break

                # ④ 所有已知任务结束 ⇒ done
                if all_done and dynamic_added:
                    yield b'data: {"event":"done"}\n\n'
                    mem.clear_session(session_id)
                    break

                # ⑤ 心跳
                if now - last_ping > 2:
                    yield b'data: {"event":"ping","message":"processing"}\n\n'
                    last_ping = now

                await asyncio.sleep(0.2)   # 主循环节拍

        except asyncio.CancelledError:
            return
        except Exception as e:
            err = {"event": "error", "detail": repr(e)}
            yield f"data: {json.dumps(err)}\n\n".encode()
            yield b'data: {"event":"done"}\n\n'

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )



if __name__ == "__main__":
    import uvicorn, json, sys, asyncio
    pipeline_recommend_params = {
        "json_file": f"{DATAFLOW_DIR}/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
        "py_path": f"{DATAFLOW_DIR}/test/recommend_pipeline.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_pipeline": False,
        "use_local_model": False,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 5
    }

    operator_write_params = {
        "json_file": f"{DATAFLOW_DIR}/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
        "py_path": f"{DATAFLOW_DIR}/test/operator_sentiment.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_operator": False,
        "use_local_model": False,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 5
    }

    if len(sys.argv) == 2 and sys.argv[1] == "recommend":
        test_req = ChatAgentRequest(
            language="zh",
            target="帮我针对数据推荐一个的pipeline!!!只需要前3个算子！！不需要去重的算子 ！",
            model="deepseek-v3",
            sessionKEY="dataflow_demo",
            **pipeline_recommend_params
        )
        resp = asyncio.run(_run_service(test_req))
        print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))
        sys.exit(0) 
    if len(sys.argv) == 2 and sys.argv[1] == "write":
        test_req = ChatAgentRequest(
            language="zh",
            target="我需要一个算子，能够对用户评论进行情感分析并输出积极/消极标签。",
            model="deepseek-v3",
            sessionKEY="dataflow_demo",
            ** operator_write_params
        )
        resp = asyncio.run(_run_service(test_req))
        print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))
        sys.exit(0)        
    uvicorn.run("run_dataflow_agent_with_ui:app", host="0.0.0.0", port=8000, reload=False)