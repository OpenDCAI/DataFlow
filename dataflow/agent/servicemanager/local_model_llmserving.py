"""
VLLM Service Serving Implementation

需求背景:
目前 dataflow Agent 的模型调用均基于 API 接口。为满足库帕斯的业务需求，现需支持本地模型部署。
因此，我们计划通过 VLLM 启动本地大模型，通过 localhost：端口的形式，实现本地模型的调用。

实现思路:
1. 现有 LocalModelLLMServing.py 中的 LocalModelLLMServing_vllm 类是直接在进程中加载 VLLM 模型
2. 需求是实现通过独立的 VLLM 服务，使用 localhost:端口 的形式调用模型
3. 创建新的 VLLMServiceServing 类，通过 HTTP API 连接到本地 VLLM 服务
4. 支持完整的 VLLM 参数配置和错误处理机制

核心特性:
- 服务模式: VLLM 作为独立服务运行，通过 HTTP API 调用
- 健康检查: 自动检测服务可用性
- 重试机制: 支持网络请求重试和错误处理
- 完整参数: 支持所有 VLLM 生成参数
- 双重功能: 支持文本生成和嵌入生成

调用方式:
1. 首先启动 VLLM 服务: python -m vllm.entrypoints.openai.api_server --model <model_path> --port 8000
2. 使用 VLLMServiceServing 类连接服务并调用模型
"""

import os
import json
import time
import requests
from typing import Optional, Union, List, Dict, Any
from dataflow import get_logger
from dataflow.core import LLMServingABC

class VLLMServiceServing(LLMServingABC):
    """
    A class for generating text using VLLM service via HTTP API.
    This class connects to a VLLM service running on localhost:port.
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 model_name: str = "default",
                 timeout: int = 30,
                 max_retries: int = 3,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1024,
                 top_k: int = 40,
                 repetition_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0,
                 seed: Optional[int] = None,
                 stop: Optional[Union[str, List[str]]] = None,
                 ):
        """
        Initialize VLLM service serving.
        
        Args:
            base_url: Base URL of the VLLM service (default: http://localhost:8000)
            api_key: API key for authentication (optional)
            model_name: Model name to use in the API calls
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            seed: Random seed for generation
            stop: Stop sequences
        """
        self.logger = get_logger()
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.stop = stop
        
        # Service status
        self.service_available = False
        self.backend_initialized = False
        
        # Setup session
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        self.logger.info(f"VLLM Service Serving initialized with base URL: {self.base_url}")
    
    def _check_service_health(self) -> bool:
        """Check if the VLLM service is available."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the VLLM service with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Request to {url} timed out after {self.timeout} seconds")
                time.sleep(1)
            
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to VLLM service at {url}")
                time.sleep(2)
            
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response: {e}")
                raise ValueError(f"Invalid JSON response from {url}")
        
        raise RuntimeError("Max retries exceeded")
    
    def start_serving(self):
        """Initialize the service connection."""
        self.backend_initialized = True
        self.logger = get_logger()
        
        # Check service health
        if not self._check_service_health():
            self.logger.warning(f"VLLM service not available at {self.base_url}")
            self.service_available = False
        else:
            self.service_available = True
            self.logger.info(f"VLLM service is available at {self.base_url}")
            
            # Try to get model info
            try:
                models_info = self._make_request("/v1/models", {})
                self.logger.info(f"Available models: {models_info}")
            except Exception as e:
                self.logger.warning(f"Could not fetch model info: {e}")
    
    def generate_from_input(self,
                           user_inputs: List[str],
                           system_prompt: str = "You are a helpful assistant") -> List[str]:
        """
        Generate responses from user inputs using VLLM service.
        
        Args:
            user_inputs: List of user input strings
            system_prompt: System prompt to use
            
        Returns:
            List of generated response strings
        """
        if not self.backend_initialized:
            self.start_serving()
        
        if not self.service_available:
            raise ConnectionError(f"VLLM service not available at {self.base_url}")
        
        results = []
        
        for user_input in user_inputs:
            # Construct messages in OpenAI format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }
            
            # Add optional parameters
            if self.seed is not None:
                payload["seed"] = self.seed
            if self.stop is not None:
                payload["stop"] = self.stop
            
            try:
                response = self._make_request("/v1/chat/completions", payload)
                
                # Extract generated text from response
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        results.append(choice["message"]["content"])
                    else:
                        results.append("")
                else:
                    self.logger.error(f"Unexpected response format: {response}")
                    results.append("")
                    
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def generate_embedding_from_input(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings from input texts using VLLM service.
        
        Args:
            texts: List of input strings
            
        Returns:
            List of embedding vectors
        """
        if not self.backend_initialized:
            self.start_serving()
        
        if not self.service_available:
            raise ConnectionError(f"VLLM service not available at {self.base_url}")
        
        results = []
        
        for text in texts:
            payload = {
                "model": self.model_name,
                "input": text,
            }
            
            try:
                response = self._make_request("/v1/embeddings", payload)
                
                # Extract embedding from response
                if "data" in response and len(response["data"]) > 0:
                    embedding = response["data"][0]["embedding"]
                    results.append(embedding)
                else:
                    self.logger.error(f"Unexpected embedding response format: {response}")
                    results.append([])
                    
            except Exception as e:
                self.logger.error(f"Error generating embedding: {e}")
                results.append([])
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        if not self.backend_initialized:
            self.start_serving()
        
        try:
            return self._make_request("/v1/models", {})
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information."""
        if not self.backend_initialized:
            self.start_serving()
        
        status = {
            "service_available": self.service_available,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "backend_initialized": self.backend_initialized,
        }
        
        if self.service_available:
            try:
                health = self.session.get(f"{self.base_url}/health", timeout=5)
                status["health_status"] = health.status_code
                status["health_response"] = health.json() if health.status_code == 200 else None
            except Exception as e:
                status["health_error"] = str(e)
        
        return status
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up VLLM Service Serving resources...")
        self.backend_initialized = False
        self.service_available = False
        
        if hasattr(self, 'session'):
            self.session.close()
        
        self.logger.info("VLLM Service Serving cleaned up")

class VLLMServiceServingOpenAI(VLLMServiceServing):
    """
    A specialized version of VLLMServiceServing that strictly follows OpenAI API format.
    """
    
    def __init__(self, **kwargs):
        # Set OpenAI-compatible defaults
        kwargs.setdefault('base_url', 'http://localhost:8000/v1')
        kwargs.setdefault('model_name', 'gpt-3.5-turbo')
        super().__init__(**kwargs)
    
    def generate_from_input(self,
                           user_inputs: List[str],
                           system_prompt: str = "You are a helpful assistant") -> List[str]:
        """
        Generate responses using strict OpenAI API format.
        """
        if not self.backend_initialized:
            self.start_serving()
        
        if not self.service_available:
            raise ConnectionError(f"VLLM service not available at {self.base_url}")
        
        # Batch process all inputs at once for efficiency
        messages_list = []
        for user_input in user_inputs:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            messages_list.append(messages)
        
        # For batch processing, we'll make individual requests
        # (VLLM doesn't support batch chat completions in the same way as OpenAI)
        results = []
        for messages in messages_list:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            
            try:
                response = self._make_request("chat/completions", payload)
                
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        results.append(choice["message"]["content"])
                    else:
                        results.append("")
                else:
                    results.append("")
                    
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                results.append(f"Error: {str(e)}")
        
        return results