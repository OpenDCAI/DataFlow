import os, asyncio
from fastapi import FastAPI
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
@app.post("/chatagent", response_model=ChatResponse)
async def chatagent(req: ChatAgentRequest):
    return await _run_service(req)

if __name__ == "__main__":
    import uvicorn, json, sys, asyncio
    pipeline_recommend_params = {
        "json_file": f"{DATAFLOW_DIR}/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
        "py_path": f"{DATAFLOW_DIR}/test/recommend_pipeline.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_pipeline": False,
        "use_local_model": True,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 5
    }

    operator_write_params = {
        "json_file": f"{DATAFLOW_DIR}/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
        "py_path": f"{DATAFLOW_DIR}/test/operator_contentSum.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_operator": False,
        "use_local_model": True,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 5
    }

    if len(sys.argv) == 2 and sys.argv[1] == "recommend":
        test_req = ChatAgentRequest(
            language="zh",
            target="帮我针对数据推荐一个的pipeline!!!不需要去重的算子 ！",
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
            target="我需要一个算子，直接使用llm_serving，实现语言翻译，把英文翻译成中文！",
            model="deepseek-v3",
            sessionKEY="dataflow_demo",
            ** operator_write_params
        )
        resp = asyncio.run(_run_service(test_req))
        print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))
        sys.exit(0)        
    uvicorn.run("test_dataflow_agent:app", host="0.0.0.0", port=8000, reload=True)

    # 我需要一个算子，能够对用户评论进行情感分析并输出积极/消极标签。
    # 我需要一个算子，能够计算文本的可读性分数并给出优化建议。
    # 我需要一个新的算子，这个算子可以使用MinHash算法进行文本去重!!
    # 我需要一个算子，直接使用LLMServing实现语言翻译，把英文翻译成中文！