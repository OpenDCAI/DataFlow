import json
import requests
import os
import logging
from ..logger import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.core  import LLMServingABC
import re

class APILLMServing_request(LLMServingABC):
    """Use OpenAI API to generate responses based on input messages.
    """
    def __init__(self, 
                 api_url: str = "https://api.openai.com/v1/chat/completions",
                 key_name_of_api_key: str = "DF_API_KEY",
                 model_name: str = "gpt-4o",
                 max_workers: int = 10
                 ):
        # Get API key from environment variable or config
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.logger = get_logger()

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key to {api_url} before using APILLMServing_request."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
  
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


    def api_chat(self, system_info: str, messages: str, model: str):
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": messages}
                ]
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

    def generate_from_input(self, 
                            user_inputs: list[str], system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        def api_chat_with_id(system_info: str, messages: str, model: str, id):
            try:
                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_info},
                        {"role": "user", "content": messages}
                    ]
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
                    return id,self.format_response(response_data)
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id,None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id,None
        responses = [None] * len(user_inputs)
        # -- end of subfunction api_chat_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    api_chat_with_id,
                    system_info = system_prompt,
                    messages = question,
                    model = self.model_name,
                    id = idx
                ) for idx, question in enumerate(user_inputs)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
    
    def generate_from_conversations(self, conversations: list[list[dict]]) -> list[str]:
        def api_chat_with_id(messages: str, model: str, id):
            try:
                payload = json.dumps({
                    "model": model,
                    "messages": messages
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
                    return id,self.format_response(response_data)
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id, None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id,None
        responses = [None] * len(conversations)
        # -- end of subfunction api_chat_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    api_chat_with_id,
                    messages = dialogue,
                    model = self.model_name,
                    id = idx
                ) for idx, dialogue in enumerate(conversations)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
    
    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        def api_embedding_with_id(text: str, model: str, id):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                data = {
                    "model": model,
                    "input": text
                }
                # Make a POST request to the API
                response = requests.post(self.api_url, headers=headers, json=data, timeout=1800)
                if response.status_code == 200:
                    # logging.info(f"API request successful")
                    response_json = response.json()
                    embedding = response_json['data'][0]['embedding']
                    # logging.info(f"API response: {response_data['choices'][0]['message']['content']}")
                    return id,embedding
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id,None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id,None
        responses = [None] * len(texts)
        # -- end of subfunction api_embedding_with_id --

        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    api_embedding_with_id,
                    text = txt,
                    model = self.model_name,
                    id = idx
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