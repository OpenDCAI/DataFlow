import os, asyncio, json, sys
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

# Configuration class - only for parameters actually used in the code
class Config:
    def __init__(self):
        # API Configuration (used in get_common_params)
        self.api_key = os.environ.get("DF_API_KEY", "")
        self.chat_api_url = os.environ.get("DF_API_URL", "")
        
        # Model Configuration (used in get_common_params)
        self.use_local_model = os.environ.get("USE_LOCAL_MODEL", "true").lower() == "true"
        self.local_model_path = os.environ.get("LOCAL_MODEL_NAME_OR_PATH", "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct")
        
        # Server Configuration (used in run_server)
        self.server_host = os.environ.get("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.environ.get("SERVER_PORT", "8000"))
        self.server_reload = os.environ.get("SERVER_RELOAD", "true").lower() == "true"
        
        # Pipeline Configuration (used in run_recommend_test)
        self.pipeline_json_file = os.environ.get("PIPELINE_JSON_FILE", "dataflow/example/ReasoningPipeline/pipeline_math_short.json")
        self.pipeline_py_path = os.environ.get("PIPELINE_PY_PATH", "test/recommend_pipeline.py")
        self.execute_pipeline = os.environ.get("EXECUTE_THE_PIPELINE", "false").lower() == "true"
        
        # Operator Configuration (used in run_write_test)
        self.operator_json_file = os.environ.get("OPERATOR_JSON_FILE", "dataflow/example/ReasoningPipeline/pipeline_math_short.json")
        self.operator_py_path = os.environ.get("OPERATOR_PY_PATH", "test/default_op.py")
        self.execute_operator = os.environ.get("EXECUTE_THE_OPERATOR", "false").lower() == "true"
        
        # Timeout Configuration (used in get_common_params)
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "3600"))
        self.max_debug_round = int(os.environ.get("MAX_DEBUG_ROUND", "5"))
        
        # Chat Agent Configuration (used in test functions)
        self.default_language = os.environ.get("DEFAULT_LANGUAGE", "zh")
        self.default_model = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
        self.default_session_key = os.environ.get("DEFAULT_SESSION_KEY", "dataflow_demo")
        
        # Test Configuration (used in test functions)
        self.test_recommend_target = os.environ.get("TEST_RECOMMEND_TARGET", "帮我针对数据推荐一个的pipeline!!!不需要去重的算子 ！")
        self.test_write_target = os.environ.get("TEST_WRITE_TARGET", "我需要一个算子，直接使用llm_serving，实现语言翻译，把英文翻译成中文！")
        
        # Debug Configuration (used in /config endpoint)
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# Initialize configuration
config = Config()

# Setup logging
logger = get_logger()

# Initialize components
toolkit = ToolRegistry()
memorys = {
    "planner": Memory(),
    "analyst": Memory(),
    "executioner": Memory(),
    "debugger": Memory(),
}

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent

def _build_task_chain(req: ChatAgentRequest, tmpl: PromptsTemplateGenerator):
    router   = TaskRegistry.get("conversation_router", prompts_template=tmpl, request=req)
    classify = TaskRegistry.get("data_content_classification", prompts_template=tmpl, request=req)
    rec      = TaskRegistry.get("recommendation_inference_pipeline", prompts_template=tmpl, request=req)
    exe      = TaskRegistry.get("execute_the_recommended_pipeline", prompts_template=tmpl, request=req)

    op_match = TaskRegistry.get("match_operator", prompts_template=tmpl, request=req)
    op_write = TaskRegistry.get("write_the_operator", prompts_template=tmpl, request=req)
    op_debug = TaskRegistry.get("exe_and_debug_operator", prompts_template=tmpl, request=req)
    task_chain = [router, classify, rec, exe, op_match, op_write, op_debug]
    cfg = build_cfg(task_chain)
    return task_chain, cfg

async def _run_service(req: ChatAgentRequest) -> ChatResponse:
    tmpl = PromptsTemplateGenerator(req.language)
    task_chain, chain_cfg = _build_task_chain(req, tmpl=tmpl)
    execution_agent = ExecutionAgent(
        request=req,
        memory_entity=memorys["executioner"],
        prompt_template=tmpl,
        debug_agent=DebugAgent(task_chain, memorys["debugger"], req),
        task_chain=task_chain
    )
    
    service = AnalysisService(
        tasks=task_chain,
        memory_entity=memorys["analyst"],
        request=req,
        execution_agent=execution_agent,
        cfg=chain_cfg
    )
    return await service.process_request()

app = FastAPI(title="Dataflow Agent Service")

@app.post("/chatagent", response_model=ChatResponse)
async def chatagent(req: ChatAgentRequest):
    return await _run_service(req)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    if not config.debug_mode:
        return {"message": "Config endpoint only available in debug mode"}
    
    return {
        "server": {
            "host": config.server_host,
            "port": config.server_port,
            "reload": config.server_reload
        },
        "model": {
            "use_local_model": config.use_local_model,
            "local_model_path": config.local_model_path,
            "default_model": config.default_model
        },
        "api": {
            "api_key_set": bool(config.api_key),
            "api_url_set": bool(config.chat_api_url)
        },
        "timeouts": {
            "request_timeout": config.request_timeout,
            "max_debug_round": config.max_debug_round
        },
        "debug_mode": config.debug_mode
    }

def get_common_params():
    return {
        "api_key": config.api_key,
        "chat_api_url": config.chat_api_url,
        "use_local_model": config.use_local_model,
        "local_model_name_or_path": config.local_model_path,
        "timeout": config.request_timeout,
        "max_debug_round": config.max_debug_round
    }

def run_recommend_test():
    pipeline_recommend_params = {
        "json_file": f"{DATAFLOW_DIR}/{config.pipeline_json_file}",
        "py_path": f"{DATAFLOW_DIR}/{config.pipeline_py_path}",
        "execute_the_pipeline": config.execute_pipeline,
        **get_common_params()
    }
    
    test_req = ChatAgentRequest(
        language=config.default_language,
        target=config.test_recommend_target,
        model=config.default_model,
        sessionKEY=config.default_session_key,
        **pipeline_recommend_params
    )
    resp = asyncio.run(_run_service(test_req))
    print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))

def run_write_test():
    operator_write_params = {
        "json_file": f"{DATAFLOW_DIR}/{config.operator_json_file}",
        "py_path": f"{DATAFLOW_DIR}/{config.operator_py_path}",
        "execute_the_operator": config.execute_operator,
        **get_common_params()
    }
    
    test_req = ChatAgentRequest(
        language=config.default_language,
        target=config.test_write_target,
        model=config.default_model,
        sessionKEY=config.default_session_key,
        **operator_write_params
    )
    resp = asyncio.run(_run_service(test_req))
    print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))

def run_server():
    import uvicorn
    uvicorn.run(
        "run_dataflow_agent_service:app",
        host=config.server_host,
        port=config.server_port,
        reload=config.server_reload
    )

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "recommend":
            run_recommend_test()
            sys.exit(0)
        elif sys.argv[1] == "write":
            run_write_test()
            sys.exit(0)
        elif sys.argv[1] == "server":
            run_server()
            sys.exit(0)
        elif sys.argv[1] == "config":
            print(json.dumps(vars(config), indent=2, default=str))
            sys.exit(0)
    
    # 默认启动服务器
    run_server()