import json
import requests
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
                 max_workers: int = 10,
                 max_retries: int = 5
                 ):
        # Get API key from environment variable or config
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.logger = get_logger()

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key to {api_url} before using APILLMServing_request."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
  
    def format_response(self, response: dict, is_embedding: bool = False) -> str:    
        # check if content is formatted like <think>...</think>...<answer>...</answer>
        if is_embedding:
            embedding = response['data'][0]['embedding']
            return embedding
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


    def api_chat(self, system_info: str, messages: str, model: str):
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": messages}
                ],
                "temperature": 0.0   
            })

            headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            # Make a POST request to the API
            response = requests.post(self.api_url, headers=headers, data=payload, timeout=60)
            if response.status_code == 200:
                response_data = response.json()
                return self.format_response(response_data)
            else:
                logging.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logging.error(f"API request error: {e}")
            return None

    def _api_chat_with_id(self, id, payload, model, is_embedding: bool = False):
            try:
                if is_embedding:
                    payload = json.dumps({
                        "model": model,
                        "input": payload
                    })
                else:
                    payload = json.dumps({
                        "model": model,
                        "messages": payload
                    })
                headers = {
                    'Authorization': f"Bearer {self.api_key}",
                    'Content-Type': 'application/json',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
                }
                # Make a POST request to the API
                response = requests.post(self.api_url, headers=headers, data=payload, timeout=1800)
                if response.status_code == 200:
                    # logging.info(f"API request successful")
                    response_data = response.json()
                    # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                    return id,self.format_response(response_data, is_embedding)
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id, None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id, None
        
    def _api_chat_id_retry(self, id, payload, model, is_embedding : bool = False):
        for i in range(self.max_retries):
            id, response = self._api_chat_with_id(id, payload, model, is_embedding)
            if response is not None:
                return id, response
            time.sleep(2**i)
        return id, None    
    
    def generate_from_input(self, 
                            user_inputs: list[str], system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:


        responses = [None] * len(user_inputs)
        # -- end of subfunction api_chat_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_id_retry,
                    payload = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                        ],
                    model = self.model_name,
                    id = idx
                ) for idx, question in enumerate(user_inputs)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
    
    def generate_from_conversations(self, conversations: list[list[dict]]) -> list[str]:

        responses = [None] * len(conversations)
        # -- end of subfunction api_chat_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_id_retry,
                    payload = dialogue,
                    model = self.model_name,
                    id = idx
                ) for idx, dialogue in enumerate(conversations)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
    
    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:

        responses = [None] * len(texts)
        # -- end of subfunction api_embedding_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_id_retry,
                    payload = txt,
                    model = self.model_name,
                    id = idx,
                    is_embedding = True
                ) for idx, txt in enumerate(texts)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embedding......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
    
    def cleanup(self):
        # Cleanup resources if needed
        logging.info("Cleaning up resources in APILLMServing_request")
        # No specific cleanup actions needed for this implementation
        pass