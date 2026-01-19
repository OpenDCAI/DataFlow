import json
import warnings
import requests
from requests.adapters import HTTPAdapter
import os
import logging
from ..logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.core  import LLMServingABC
import re
import time

class APILLMServing_request(LLMServingABC):
    """Use OpenAI API to generate responses based on input messages.
    """
    def start_serving(self) -> None:
        self.logger.info("APILLMServing_request: no local service to start.")
        return
    
    def __init__(self, 
                 api_url: str = "https://api.openai.com/v1/chat/completions",
                 key_name_of_api_key: str = "DF_API_KEY",
                 model_name: str = "gpt-4o",
                 temperature: float = 0.0,
                 max_workers: int = 10,
                 max_retries: int = 5,
                 connect_timeout: float = 10.0,
                 read_timeout: float = 120.0,
                 **configs : dict
                 ):
        # Get API key from environment variable or config
        self.api_url = api_url
        self.model_name = model_name
        # self.temperature = temperature
        self.max_workers = max_workers
        self.max_retries = max_retries

        self.timeout = (connect_timeout, read_timeout)  # (connect timeout, read timeout)
        # check for deprecated timeout
        if 'timeout' in configs:
            warnings.warn("The `timeout` parameter is deprecated. Please use `connect_timeout` and `read_timeout` instead.", DeprecationWarning)
            self.timeout = (connect_timeout, configs['timeout'])
            configs.pop('timeout')

        self.configs = configs
        self.configs.update({"temperature": temperature})

        self.logger = get_logger()

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key to {api_url} before using APILLMServing_request."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        

        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers,
            max_retries=0,  # 你已经有 _api_chat_id_retry 了，这里不要重复重试
            pool_block=True # 池满时阻塞，避免无限建连接导致资源抖动
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
  
        self.headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
        }


    def format_response(self, response: dict, is_embedding: bool = False) -> str:
        """Format API response, supporting both embedding and chat completion modes"""
        
        # Handle embedding requests
        if is_embedding:
            return response.get('data', [{}])[0].get('embedding', [])
        
        # Extract message content
        message = response.get('choices', [{}])[0].get('message', {})
        content = message.get('content', '')
        
        # Return directly if content is already in think/answer format
        if re.search(r'<think>.*?</think>.*?<answer>.*?</answer>', content, re.DOTALL):
            return content
        
        # Check for reasoning_content
        reasoning_content = message.get('reasoning_content')
        
        # Wrap with think/answer tags if reasoning_content exists and is not empty
        if reasoning_content:
            return f"<think>{reasoning_content}</think>\n<answer>{content}</answer>"
        
        return content
    # deprecated
    # def api_chat(self, system_info: str, messages: str, model: str):
    #     try:
    #         payload = json.dumps({
    #             "model": model,
    #             "messages": [
    #                 {"role": "system", "content": system_info},
    #                 {"role": "user", "content": messages}
    #             ],
    #             "temperature": self.temperature   
    #         })

    #         headers = {
    #             'Authorization': f"Bearer {self.api_key}",
    #             'Content-Type': 'application/json',
    #             'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    #         }
    #         # Make a POST request to the API
    #         response = requests.post(self.api_url, headers=headers, data=payload, timeout=60)
    #         if response.status_code == 200:
    #             response_data = response.json()
    #             return self.format_response(response_data)
    #         else:
    #             logging.error(f"API request failed with status {response.status_code}: {response.text}")
    #             return None
    #     except Exception as e:
    #         logging.error(f"API request error: {e}")
    #         return None

    def _api_chat_with_id(
              self, 
              id: int, 
              payload, 
              model: str, 
              is_embedding: bool = False, 
              json_schema: dict = None
              ):
            start = time.time()
            try:
                if is_embedding:
                    payload = {
                        "model": model,
                        "input": payload
                    }
                elif json_schema is None:
                    payload = {
                        "model": model,
                        "messages": payload
                    }
                else:
                    payload = {
                        "model": model,
                        "messages": payload,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "custom_response",
                                "strict": True,
                                "schema": json_schema
                            }
                        }
                    }
                    
                payload.update(self.configs)
                payload = json.dumps(payload)
                # Make a POST request to the API
                response = self.session.post(self.api_url, headers=self.headers, data=payload, timeout=self.timeout)
                cost = time.time() - start
                if response.status_code == 200:
                    # logging.info(f"API request successful")
                    response_data = response.json()
                    # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                    return id,self.format_response(response_data, is_embedding)
                else:
                    # self.logger.exception(f"API request failed (id = {id}) with status {response.status_code}: {response.text}")
                    self.logger.error(f"API request failed id={id} status={response.status_code} cost={cost:.2f}s body={response.text[:500]}")
                    return id, None
            # ✅ 1) 连接阶段超时：认为“根本连不上” => 统一抛 RuntimeError（Win/Linux 一致）
            except requests.exceptions.ConnectTimeout as e:
                cost = time.time() - start
                self.logger.error(f"API connect timeout (id={id}) cost={cost:.2f}s: {e}")
                raise RuntimeError(f"Cannot connect to LLM server (connect timeout): {e}") from e

            # ✅ 2) 读超时：服务可达但没有数据（排队/推理太久）=> warn + None
            except requests.exceptions.ReadTimeout as e:
                cost = time.time() - start
                warnings.warn(f"API read timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning)
                return id, None

            # ✅ 3) 其他 Timeout（极少）按 warn 处理
            except requests.exceptions.Timeout as e:
                cost = time.time() - start
                warnings.warn(f"API timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning)
                return id, None

            # ✅ 4) ConnectionError：这里面 Win/Linux/urllib3 可能包装了各种“超时/断开”
            except requests.exceptions.ConnectionError as e:
                cost = time.time() - start
                msg = str(e).lower()

                # requests/urllib3 有时会把 ReadTimeout 包装成 ConnectionError
                if "read timed out" in msg:
                    warnings.warn(f"API read timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning)
                    return id, None

                # 连接阶段超时在某些平台也可能表现为 ConnectionError 文本
                if "connect timeout" in msg or ("timed out" in msg and "connect" in msg):
                    self.logger.error(f"API connect timeout (id={id}) cost={cost:.2f}s: {e}")
                    raise RuntimeError(f"Cannot connect to LLM server (connect timeout): {e}") from e

                # 其它连接错误：refused/reset/remote disconnected 等 => 统一抛 RuntimeError
                self.logger.error(f"API connection error (id={id}) cost={cost:.2f}s: {e}")
                raise RuntimeError(f"Cannot connect to LLM server: {e}") from e

            except Exception as e:
                cost = time.time() - start
                self.logger.exception(f"API request error (id = {id}) cost={cost:.2f}s: {e}")
                return id, None
        
    def _api_chat_id_retry(self, id, payload, model, is_embedding : bool = False, json_schema: dict = None):
        for i in range(self.max_retries):
            id, response = self._api_chat_with_id(id, payload, model, is_embedding, json_schema)
            if response is not None:
                return id, response
            time.sleep(2**i)
        return id, None    
    
    def _run_threadpool(self, task_args_list: list[dict], desc: str) -> list:
        """
        task_args_list: 每个元素都是 _api_chat_id_retry 的入参 dict
        e.g. {"id": 0, "payload": [...], "model": "...", "is_embedding": False, "json_schema": None}
        返回值按 id 回填到 responses
        """
        responses = [None] * len(task_args_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._api_chat_id_retry, **task_args)
                for task_args in task_args_list
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    response = future.result()  # (id, response)
                    # response[0] 是 id，response[1] 是实际响应， 用于按 id 回填 responses 列表
                    responses[response[0]] = response[1]
                except Exception:
                    # 理论上 worker 内部已经 try/except 了，但这里兜底更安全
                    self.logger.exception("Worker crashed unexpectedly in threadpool")

        return responses
    
    def generate_from_input(self, 
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant",
                            json_schema: dict = None,
                            ) -> list[str]:
        task_args_list = [
            dict(
                id=idx,
                payload=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                model=self.model_name,
                json_schema=json_schema,
            )
            for idx, question in enumerate(user_inputs)
        ]
        return self._run_threadpool(task_args_list, desc="Generating responses from prompts......")

    
    def generate_from_conversations(self, conversations: list[list[dict]]) -> list[str]:

        task_args_list = [
            dict(
                id=idx,
                payload=dialogue,
                model=self.model_name,
            )
            for idx, dialogue in enumerate(conversations)
        ]
        return self._run_threadpool(task_args_list, desc="Generating responses from conversations......")
              
    
    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        task_args_list = [
            dict(
                id=idx,
                payload=txt,
                model=self.model_name,
                is_embedding=True,
            )
            for idx, txt in enumerate(texts)
        ]
        return self._run_threadpool(task_args_list, desc="Generating embedding......")
    
    def cleanup(self):
        self.logger.info("Cleaning up resources in APILLMServing_request")
        try:
            if hasattr(self, "session") and self.session:
                self.session.close()
        except Exception:
            self.logger.exception("Failed to close requests session")