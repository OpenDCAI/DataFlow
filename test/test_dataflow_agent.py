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
        "py_path": f"{DATAFLOW_DIR}/test/recommend_pipeline_test.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_pipeline": False,
        "use_local_model": False,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 20
    }
# /mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl
    operator_write_params = {
        "json_file": f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        "py_path": f"{DATAFLOW_DIR}/test/op_test_deepseek-v3.py",
        "api_key": api_key,
        "chat_api_url": chat_api_url,
        "execute_the_operator": True,
        "use_local_model": False,
        "local_model_name_or_path": "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        "timeout": 3600,
        "max_debug_round": 5
    }

    if len(sys.argv) == 2 and sys.argv[1] == "recommend":
        test_req = ChatAgentRequest(
            language="zh",
            target="帮我针对数据推荐一个pipeline!!!",
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
            target="我需要一个算子，使用LLMServing对医疗场景的原始题干进行同义改写，生成语义一致但表达不同的新问题，有效增加训练样本多样性，并且输入key是question，输出key是questionPARA,就在原数据上新加入key，不要生成新的行。",
            model="gpt-4o-mini",
            sessionKEY="dataflow_demo",
            ** operator_write_params
        )
        resp = asyncio.run(_run_service(test_req))
        print(json.dumps(resp.dict(), ensure_ascii=False, indent=2))
        sys.exit(0)        
    uvicorn.run("test_dataflow_agent:app", host="0.0.0.0", port=8000, reload=True)

    # 我需要一个新的算子，这个算子可以使用MinHash算法进行文本去重!!    
    # 我需要一个算子，能够检测文本中是否包含代码片段或数学公式，并进行格式化。
    # 我需要一个算子，能够自动从文本中提取最能表达主题的关键词。
    # 我需要一个算子，能够检测文本长度和密度。
    # 我需要一个算子，直接使用LLMServing实现语言翻译，把英文翻译成中文！
    # 我需要一个算子，能够通过LLM识别文本中的人名、地名、机构名等命名实体。
    # 我需要一个算子，能够根据文本自动生成相关问题。
    # 我需要一个算子，能够对用户评论进行情感分析并输出积极/消极标签。
    # 我需要一个算子，能够通过LLM自动识别并过滤垃圾广告或灌水内容。
    # 我需要一个算子，直接使用LLMservinf进行总结，能够对输入的长文本自动生成摘要。
    # 我需要一个算子，能够根据给定主题或关键词自动生成新文本。


# 医学："我需要一个算子，使用LLMServing对医疗场景的原始题干进行同义改写，生成语义一致但表达不同的新问题，有效增加训练样本多样性，并且输入key是question，输出key是questionPARA,就在原数据上新加入key，不要生成新的行。",
# 我需要一个算子，使用LLMServing对医疗场景的原始题干进行临床细节扩充，在不影响考点的前提下，插入合理的病史、体格检查或辅助检查结果，增强问题的真实性，并可灵活调节题目难度。输入key是question，输出key是questionCONTEXT。
# 我需要一个算子，使用LLMServing针对医疗选择题，自动生成从题干到标准答案的推理过程（思维链/Chain-of-Thought），以提升模型的推理能力和可解释性。输入key是question，输出key是questionCOT，内容为详细的推理步骤。
# 我需要一个算子，使用LLMServing针对医疗多项选择题，自动生成具有医学迷惑性的干扰选项（错误选项），让错误选项更“像真”，提升模型判别难度。需依据常见临床误区或相近伦理概念设计，输入key是options，输出key是distractorGEN。
