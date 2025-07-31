import os
import socket
import subprocess
import time
import signal
import logging
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.INFO
)
from sglang.test.test_utils import is_in_ci
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process

class SGLangServer:
    """
    Robust manager for a local sglang_router server process,
    with real-time log capture and health checks.
    """

    def __init__(
        self,
        model_path: str,
        tp: int = 1,
        max_total_tokens: int = 2048,
        max_workers: int = 10,
        temperature: float = 0.7,
    ):
        """
        Initialize the server manager.

        Args:
            model_path: HuggingFace Hub path or local model directory.
            dp: data parallel degree.
            tp: tensor parallel degree.
            max_total_tokens: maximum tokens to generate per request.
            host: bind address for the server.
            startup_timeout: seconds to wait for server to become healthy.
            server_args: additional flags passed to the sglang_router server.
        """
        self.model_path = model_path
        self.tp = tp
        self.max_total_tokens = max_total_tokens
        self.max_workers = max_workers
        self.temperature = temperature
        self.url = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    def find_free_port(self) -> int:
        """自动查找系统空闲端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


    def launch_server(self) -> str:
        """
        Launch the sglang_router server process, and wait until it's healthy.
        """
        server_process, port = launch_server_cmd(
            f"python3 -m sglang.launch_server --model-path {self.model_path} --host 0.0.0.0 --tp {self.tp} --port {self.find_free_port()}"
        )
        logging.info(f"Waiting for server to become healthy...")
        wait_for_server(f"http://localhost:{port}")
        self.url = f"http://localhost:{port}/generate"
        logging.info(f"Server running at {self.url}")
        return f"http://localhost:{port}/generate"

        
    def generate_from_input(
        self,
        user_inputs: List[str],
        system_prompt: str = "You are a helpful assistant",
    ) -> List[str]:
        """
        Generate responses for each user input via the local server.

        Args:
            user_inputs: list of user query strings.
            system_prompt: system message prefix for each conversation.

        Returns:
            list of response strings.
        """
        if not self.url:
            self.launch_server()

        responses = [None] * len(user_inputs)
        def get_response_with_id(text, id):
            response = requests.post(
                self.url,
                json={
                    "text": text,
                    "sampling_params": {
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_total_tokens,
                    },
                },
            )
            # print(response.json())
            return response.json()["text"], id
        fp = []
        for question in user_inputs:
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
            fp.append(messages)
        ft = self.tokenizer.apply_chat_template(
            fp,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(get_response_with_id, text, i) for i, text in enumerate(ft)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                response, id = future.result()
                responses[id] = response
        return responses
        



if __name__ == '__main__':
    server = SGLangServer(
        model_path="/data0/public_models/Qwen2.5-7B-Instruct",
        tp=2,
        max_total_tokens=2048,
        max_workers=10,
        temperature=0.7,
    )
    server.launch_server()
    print(f"Server running at {server.url}")
    responses = server.generate_from_input(
        ["介绍一下你自己","介绍一下Qwen"],
        system_prompt="You are a helpful assistant",
    )
    print(responses)
