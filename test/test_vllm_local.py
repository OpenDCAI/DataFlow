#!/usr/bin/env python3
"""
测试脚本：使用本地GPU通过localhost调用模型进行推理
"""

import os
import sys
import torch
import uvicorn
import subprocess
import time
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dataflow.agent.servicemanager.local_model_llmserving import VLLMServiceServing
    from dataflow import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行此脚本")
    sys.exit(1)

# 定义API请求和响应模型
class ChatRequest(BaseModel):
    user_inputs: List[str]
    system_prompt: str = "You are a helpful assistant"

class ChatResponse(BaseModel):
    responses: List[str]
    service_url: str
    model_name: str

def start_vllm_server():
    """启动vLLM服务器进程"""
    global vllm_process
    logger = get_logger()
    
    # 模型路径
    model_path = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"
    port = 12345
    host = "127.0.0.1"
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False
    
    # 构建启动命令
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "2048",
        "--dtype", "auto"
    ]
    
    logger.info(f"启动vLLM服务器: {' '.join(command)}")
    
    try:
        # 启动服务器
        vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        
        logger.info(f"vLLM服务器启动中... (PID: {vllm_process.pid})")
        
        # 等待服务器启动
        for i in range(60):  # 最多等待60秒
            try:
                response = requests.get(f"http://{host}:{port}/v1/models", timeout=2)
                if response.status_code == 200:
                    logger.info("✓ vLLM服务器启动成功！")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            
        logger.error("vLLM服务器启动超时")
        return False
        
    except Exception as e:
        logger.error(f"启动vLLM服务器失败: {e}")
        return False

def start_vllm_service():
    """启动vLLM服务并返回服务实例"""
    global vllm_service
    logger = get_logger()
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法运行GPU推理")
        return None
    
    logger.info(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
    logger.info(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 模型路径
    model_path = "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct"
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return None
    
    logger.info(f"使用模型: {model_path}")
    
    # 启动vLLM服务器
    if not start_vllm_server():
        logger.error("启动vLLM服务器失败")
        return None
    
    # 创建vLLM服务实例 - 使用VLLMServiceServing
    try:
        logger.info("创建vLLM服务实例...")
        vllm_service = VLLMServiceServing(
            base_url="http://127.0.0.1:12345",
            model_name="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
            timeout=30,
            max_retries=3,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
        logger.info("vLLM服务实例创建成功")
    except Exception as e:
        logger.error(f"创建vLLM服务实例失败: {e}")
        return None
    
    # 启动服务连接
    try:
        logger.info("连接到vLLM服务器...")
        vllm_service.start_serving()
        logger.info("vLLM服务连接成功")
        return vllm_service
    except Exception as e:
        logger.error(f"连接vLLM服务器失败: {e}")
        return None

def test_vllm_serving():
    """测试vLLM服务功能"""
    logger = get_logger()
    
    # 启动vLLM服务
    vllm_service = start_vllm_service()
    if vllm_service is None:
        return False
    
    # 测试推理
    try:
        logger.info("开始测试推理...")
        
        # 测试问题
        test_questions = [
            "你好，请简单介绍一下你自己。",
            "什么是机器学习？",
            "请用一句话总结Python的优点。"
        ]
        
        logger.info(f"测试问题: {test_questions}")
        
        # 进行推理
        responses = vllm_service.generate_from_input(
            user_inputs=test_questions,
            system_prompt="你是一个有帮助的AI助手，请用中文回答问题。"
        )
        
        logger.info("推理完成，结果如下:")
        for i, (question, response) in enumerate(zip(test_questions, responses)):
            logger.info(f"\n问题 {i+1}: {question}")
            logger.info(f"回答 {i+1}: {response}")
            logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"推理测试失败: {e}")
        return False

# 全局变量存储vLLM服务实例和进程
vllm_service = None
vllm_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vllm_service
    logger = get_logger()
    
    # 启动时初始化vLLM服务
    logger.info("应用启动，初始化vLLM服务...")
    vllm_service = start_vllm_service()
    if vllm_service is None:
        logger.error("vLLM服务启动失败")
        raise RuntimeError("vLLM服务启动失败")
    logger.info("vLLM服务启动成功")
    
    yield
    
    # 关闭时清理资源
    logger.info("应用关闭，清理资源...")
    if vllm_service:
        try:
            vllm_service.cleanup()
            logger.info("vLLM服务清理完成")
        except Exception as e:
            logger.error(f"清理vLLM服务失败: {e}")
    
    if vllm_process:
        try:
            import signal
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
            vllm_process.wait()
            logger.info("vLLM服务器进程已停止")
        except Exception as e:
            logger.error(f"停止vLLM服务器进程失败: {e}")

# 创建FastAPI应用
app = FastAPI(title="VLLM Local Service", lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天接口"""
    global vllm_service
    
    if vllm_service is None:
        return ChatResponse(
            responses=["vLLM服务未启动"],
            service_url="",
            model_name=""
        )
    
    try:
        # 生成回复
        responses = vllm_service.generate_from_input(
            user_inputs=request.user_inputs,
            system_prompt=request.system_prompt
        )
        
        # 构建服务URL
        service_url = vllm_service.base_url
        
        return ChatResponse(
            responses=responses,
            service_url=service_url,
            model_name=vllm_service.model_name
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"聊天接口调用失败: {e}")
        return ChatResponse(
            responses=[f"错误: {str(e)}"],
            service_url="",
            model_name=""
        )

@app.get("/service_info")
async def get_service_info():
    """获取服务信息"""
    global vllm_service
    
    if vllm_service is None:
        return {
            "service_available": False,
            "message": "vLLM服务未启动"
        }
    
    return {
        "service_available": True,
        "service_url": vllm_service.base_url,
        "model_name": vllm_service.model_name,
        "host": vllm_service.base_url.split(":")[1].strip("/"),
        "port": vllm_service.base_url.split(":")[-1],
        "backend_initialized": vllm_service.backend_initialized
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    global vllm_service
    
    if vllm_service is None or not vllm_service.backend_initialized:
        return {"status": "unhealthy", "message": "vLLM服务未启动"}
    
    return {"status": "healthy", "message": "vLLM服务正常运行"}

def main():
    """主函数"""
    logger = get_logger()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # 运行测试模式
            logger.info("开始vLLM本地推理测试")
            success = test_vllm_serving()
            
            if success:
                logger.info("测试成功完成！")
                sys.exit(0)
            else:
                logger.error("测试失败！")
                sys.exit(1)
        elif sys.argv[1] == "server":
            # 启动服务器模式
            logger.info("启动vLLM服务模式")
            logger.info("服务启动后，可以通过以下方式访问:")
            logger.info("- 健康检查: http://localhost:8001/health")
            logger.info("- 服务信息: http://localhost:8001/service_info")
            logger.info("- 聊天接口: http://localhost:8001/chat")
            logger.info("- API文档: http://localhost:8001/docs")
            logger.info("DFA可以通过 http://localhost:8001/service_info 获取服务URL")
            
            # 启动FastAPI服务器
            uvicorn.run("test_vllm_local:app", host="0.0.0.0", port=8001, reload=False)
        else:
            logger.error("未知参数，使用 'test' 运行测试或 'server' 启动服务")
            sys.exit(1)
    else:
        # 默认启动服务器模式
        logger.info("启动vLLM服务模式")
        logger.info("服务启动后，可以通过以下方式访问:")
        logger.info("- 健康检查: http://localhost:8001/health")
        logger.info("- 服务信息: http://localhost:8001/service_info")
        logger.info("- 聊天接口: http://localhost:8001/chat")
        logger.info("- API文档: http://localhost:8001/docs")
        logger.info("DFA可以通过 http://localhost:8001/service_info 获取服务URL")
        
        # 启动FastAPI服务器
        uvicorn.run("test_vllm_local:app", host="0.0.0.0", port=8001, reload=False)

if __name__ == "__main__":
    main()