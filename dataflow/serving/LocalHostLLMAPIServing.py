import subprocess
import signal
import os
import re
import time
import requests
from threading import Thread
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import LLMServingABC
from concurrent.futures import ThreadPoolExecutor, as_completed


class LocalHostLLMAPIServing_vllm(LLMServingABC):
    """
    A class to serve vLLM via a subprocess (e.g., localhost API server)
    """
    def __init__(self,
                 hf_model_name_or_path: str,
                 port: int = 12345,
                 host: str = "127.0.0.1",
                 tensor_parallel_size: int = 1,
                 max_model_len: int = None,
                 gpu_memory_utilization: float = 0.9,
                 hf_cache_dir: str = None,
                 server_start_timeout: int = 120,
                 max_workers: int = 16):
        self.logger = get_logger()
        self.hf_model_name_or_path = hf_model_name_or_path
        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_cache_dir = hf_cache_dir
        self.server_start_timeout = server_start_timeout
        self.max_workers = max_workers
        self.process = None
        self.backend_initialized = False

    def start_serving(self):
        if self.backend_initialized:
            self.logger.info("vLLM server already running.")
            return

        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.hf_model_name_or_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        if self.max_model_len:
            command += ["--max-model-len", str(self.max_model_len)]
        if self.hf_cache_dir:
            command += ["--download-dir", self.hf_cache_dir]

        self.logger.info(f"Starting vLLM server with command: {' '.join(command)}")
        self.process = subprocess.Popen(command, preexec_fn=os.setsid)

        for i in range(self.server_start_timeout):  # 增加等待时间
            try:
                response = requests.get(f"http://{self.host}:{self.port}/v1/models", timeout=1.0)
                status = response.status_code
                self.logger.debug(f"[{i+1}/90] Status: {status}")
                if status == 200:
                    self.backend_initialized = True
                    self.logger.success("vLLM server started successfully!")
                    return
            except Exception as e:
                self.logger.debug(f"[{i+1}/90] Connection failed: {repr(e)}")
            time.sleep(1)

        self.cleanup()
        raise RuntimeError("Failed to start vLLM server within timeout.")

    def format_response(self, response: dict) -> str:    
        # check if content is formatted like <think>...</think>...<answer>...</answer>
        content = response['choices'][0]['message']['content']
        if re.search(r'<think>.*</think>.*<answer>.*</answer>', content):
            return content
        
        try:
            reasoning_content = response['choices'][0]["message"]["reasoning_content"]
        except:
            reasoning_content = ""
        
        if reasoning_content != "":
            return f"<think>{reasoning_content}</think>\n<answer>{content}</answer>"
        else:
            return content

    def _api_chat_with_id(self, id, payload, model):
            try:
                payload = {
                    "model": self.hf_model_name_or_path,
                    "messages": payload,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }

                response = requests.post(f"http://{self.host}:{self.port}/v1/chat/completions", json=payload)
                if response.status_code == 200:
                    # logging.info(f"API request successful")
                    response_data = response.json()
                    # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                    return id,self.format_response(response_data)
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id, None
            except Exception as e:
                self.logger.error(f"API request error: {e}")
                return id, None
    
    def generate_from_input(self, 
                            user_inputs: list[str], system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:

        if not self.backend_initialized:
            self.start_serving()
        responses = [None] * len(user_inputs)

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_with_id,
                    payload = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                        ],
                    model = self.hf_model_name_or_path,
                    id = idx
                ) for idx, question in enumerate(user_inputs)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses

    # def generate_from_input(self, user_inputs: list[str], system_prompt: str = "You are a helpful assistant") -> list[str]:
    #     if not self.backend_initialized:
    #         self.start_serving()

    #     messages = [{"role": "system", "content": system_prompt}] + \
    #                [{"role": "user", "content": q} for q in user_inputs]

    #     payload = {
    #         "model": self.hf_model_name_or_path,
    #         "messages": messages,
    #         "temperature": 0.7,
    #         "top_p": 0.9,
    #         "max_tokens": 1024
    #     }

    #     response = requests.post(f"http://{self.host}:{self.port}/v1/chat/completions", json=payload)
    #     response.raise_for_status()
    #     data = response.json()
    #     return [choice["message"]["content"] for choice in data["choices"]]

    def cleanup(self):
        if self.process:
            self.logger.info("Shutting down vLLM subprocess...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            time.sleep(10)
            self.logger.success("vLLM subprocess terminated.")
            self.backend_initialized = False