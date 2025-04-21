import json
import requests
import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tinydb import TinyDB

class APIGenerator_request:
    def __init__(self, config: dict):
        self.config = config
        self.check_config()
        self.db = TinyDB(self.config['db_path'])
        logging.info(f"DB path: {self.config['db_path']}")
        
        # Get API key from environment variable or config
        self.api_url = self.config.get("api_url", None)
        self.api_key = self.config.get("api_key", None)
        if self.api_key == "":
            self.api_key = os.environ.get("API_KEY")
        if self.api_key is None:
            raise ValueError("Lack of API_KEY")
    
    def check_config(self):
        # Ensure all necessary keys are in the config
        necessary_keys = ['db_path', 'api_url', 'input_file', 'output_file', 'input_key', 'output_key', 'max_workers']
        for key in necessary_keys:
            if key not in self.config:
                raise ValueError(f"Key {key} is missing in the config")

    def api_chat(self, system_info: str, messages: str, model: str, finish_try: int = 3):
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
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            else:
                logging.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logging.error(f"API request error: {e}")
            return None

    def generate_text(self):
        # Read input file (jsonl only)
        raw_dataframe = pd.read_json(self.config['input_file'], lines=True)
        if self.config['input_key'] not in raw_dataframe.columns:
            raise ValueError(f"input_key: {self.config['input_key']} not found in the dataframe.")
        
        logging.info(f"Found {len(raw_dataframe)} rows in the dataframe")
        
        # Create a copy for parallel processing
        dataframe = raw_dataframe.copy()
        dataframe['id'] = dataframe.index

        # Load existing IDs from DB
        existing_ids = [item['id'] for item in self.db.all()]
        # Filter out existing IDs
        dataframe = dataframe[~dataframe['id'].isin(existing_ids)]
        logging.info(f"Found {len(dataframe)} rows to generate responses.")

        responses = [None] * len(dataframe)  # 创建一个列表，确保结果顺序与输入数据一致

        # Use ThreadPoolExecutor to handle multiple requests concurrently
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            for idx, row in dataframe.iterrows():
                futures.append(
                    executor.submit(
                        self.api_chat,
                        self.config['system_prompt'],
                        row[self.config['input_key']],
                        self.config['model_name'],
                    )
                )
            
            for idx, future in enumerate(as_completed(futures)):
                response = future.result()
                responses[idx] = response  # 将响应放到正确的索引位置，确保顺序一致

        # Save the responses into the TinyDB
        for idx, response in enumerate(responses):
            raw_input = dataframe.loc[idx, self.config['input_key']]
            self.db.insert({
                'id': dataframe.loc[idx, 'id'],
                self.config['input_key']: raw_input,
                'response': response
            })
            logging.info(f"Saved response for id {dataframe.loc[idx, 'id']} to DB")
        
        outputs = [None] * len(dataframe)
        for item in self.db.all():
            outputs[item["id"]] = item["response"]
        return outputs

    def generate_and_save(self):
        # Generate text and save to file
        responses = self.generate_text()
        raw_dataframe = pd.read_json(self.config['input_file'], lines=True)
        raw_dataframe[self.config['output_key']] = responses
        raw_dataframe.to_json(self.config['output_file'], orient='records', lines=True, force_ascii=False)

    def generate_text_from_input(self, questions: list[str]) -> list[str]:
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
                    return id,response_data['choices'][0]['message']['content']
                else:
                    logging.error(f"API request failed with status {response.status_code}: {response.text}")
                    return id,None
            except Exception as e:
                logging.error(f"API request error: {e}")
                return id,None
        responses = [None] * len(questions)
        
        # 使用 ThreadPoolExecutor 并行处理多个问题
        # logging.info(f"Generating {len(questions)} responses")
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = [executor.submit(
                        api_chat_with_id,
                        self.config['system_prompt'],
                        question,
                        self.config['model_name'],
                        idx
                    ) for idx, question in enumerate(questions)]
            
            for future in as_completed(futures):
                response = future.result()
                responses[response[0]] = response[1]
                # for debug
                # logging.info(f"Get {response[0]} responses")
        # logging.info(f"Generated {len(responses)} responses")
        return responses
