# A special algotithm for generating answers with a reasoning model
import json
from aiohttp import Payload
import requests
import threading
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tinydb import TinyDB
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger

@GENERATOR_REGISTRY.register()
class AnswerGenerator_reasoning:
    '''
    For QwQ-32B and Deepseek-R1
    '''
    def __init__(self, config : dict):
        self.config = config
        self.check_config()
        self.db = TinyDB(self.config['db_path'])
        self.logger = get_logger()
        self.logger.info(f"DB path: {self.config['db_path']}")
        self.max_workers = self.config.get('max_workers',4)
        self.output_content_key = self.config.get('output_content_key', 'content')
        self.output_reasoning_key = self.config.get('output_reasoning_key', 'reasoning_content')
        self.output_total_token_key = self.config.get('output_total_token_key', 'total_token')

    def check_config(self):
        # check if necessary keys are in the config
        necessary_keys = ['db_path',
                          'system_prompt',
                          'input_key',
                          'model_name',
                          'url',
                          'api_key',
                          'output_file',
                          'input_file',
                          'max_workers'
                          ]
        for key in necessary_keys:
            if key not in self.config:
                raise ValueError(f"Key {key} is not in the config")

    def chat(self,system_prompt,message,model_name,url,api_key,id):
        try:
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            })
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            response = requests.post(url, data=payload, headers=headers,timeout=1800)
            status_code = response.status_code
            if status_code == 200:
                self.logger.info(f"Response code is 200, Get the answer successfully")
                return response.json(),id
            else:
                self.logger.error(f"Error: {status_code} - {response.text}")
                return None,id
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None,id

    def Analyze_response_json(self, response_json):
        '''
        Analyze the response json and return the answer
        '''
        # check stop reason
        if response_json["choices"][0]["finish_reason"] != "stop":
            self.logger.error(f"Error: The model stopped for reason: {response_json['choices'][0]['finish_reason']}")
            return None
        
        # check if reasoning_content exists
        if 'reasoning_content' in response_json['choices'][0]['message'] and response_json['choices'][0]['message']['reasoning_content'] != "":
            self.logger.info(f"Get reasoning content successfully")
            reasoning_content = response_json['choices'][0]['message']['reasoning_content']
            content = response_json['choices'][0]['message']['content']
        else:
            self.logger.info(f"No reasoning content, try to parse reasoning part in content")
            text = response_json['choices'][0]['message']['content']
            text_split = text.split("</think>")
            if len(text_split) == 2 and "<think>" in text_split[0]:
                reasoning_content = text_split[0]
                content = text_split[1]
                # remove <think> and <answer> and </think> and </answer>
                reasoning_content = reasoning_content.replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
                content = content.replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
            else:
                self.logger.error(f"Error: Failed to parse reasoning content from the response")
                return None
        total_token = response_json['usage']['total_tokens']
        return reasoning_content, content, total_token
    
    def save_db_to_file(self):
        '''
        Save the db to file
        '''
        raw_dataframe = pd.read_json(self.input_file,lines=True)
        for item in self.db.all():
            raw_dataframe.loc[item['id'],"content"] = item['content']
            raw_dataframe.loc[item['id'],"total_token"] = item['total_token']
            raw_dataframe.loc[item['id'],"reasoning_content"] = item['reasoning_content']
        raw_dataframe.to_json(self.output_file,orient='records',lines=True,force_ascii=False)

    
    def run(self):
        '''
        Run the algorithm
        '''
        # read input file : accept jsonl file only
        raw_dataframe = pd.read_json(self.config['input_file'],lines=True)
        # check if input_key are in the dataframe
        if self.config['input_key'] not in raw_dataframe.columns:
            key_list = raw_dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.config['input_key']} not found in the dataframe, please check the input_key: {key_list}")
        self.logger.info(f"Found {len(raw_dataframe)} rows in the dataframe")
        # generate id for hash
        dataframe = raw_dataframe.copy()
        dataframe['id'] = dataframe.index

        # load existing ids from db
        existing_ids = [item['id'] for item in self.db.all()]
        # filter out existing ids
        dataframe = dataframe[~dataframe['id'].isin(existing_ids)]
        self.logger.info(f"Found {len(existing_ids)} existing ids, there are {len(dataframe)} rows to generate")

        # generate answer and save at once
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            for index, row in dataframe.iterrows():
                futures.append(
                    executor.submit(
                        self.chat,
                        self.config['system_prompt'],
                        row[self.config['input_key']],
                        self.config['model_name'],
                        self.config['url'],
                        self.config['api_key'],
                        row['id']
                    )
                )
                self.logger.info(f"Submitted task {index} of {len(dataframe)}")
            for future in as_completed(futures):
                response_json,id = future.result()
                reasoning_content, content, total_token = self.Analyze_response_json(response_json)
                raw_input = dataframe.loc[id,self.config['input_key']]
                self.db.insert({
                    'id':id,
                    self.config['input_key']:raw_input,
                    self.output_reasoning_key:reasoning_content,
                    self.output_content_key:content,
                    self.output_total_token_key:total_token
                })
                self.logger.info(f"Saved {id} to db")
        
        # save dataframe to file
        for item in self.db.all():
            raw_dataframe.loc[item['id'],self.output_content_key] = item[self.output_content_key]
            raw_dataframe.loc[item['id'],self.output_total_token_key] = item[self.output_total_token_key]
            raw_dataframe.loc[item['id'],self.output_reasoning_key] = item[self.output_reasoning_key]
        raw_dataframe.to_json(self.config['output_file'],orient='records',lines=True)



                