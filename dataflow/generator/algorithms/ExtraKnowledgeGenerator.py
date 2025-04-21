from typing import Dict, List, Optional, Union
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
from dataflow.generator.utils.Prompts import ExtraKnowledgePrompt
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class ExtraKnowledgeGenerator:
    def __init__(self, config: Dict):
        self.config = config
        
        self.prompt = ExtraKnowledgePrompt()
        self._lock = threading.Lock() 
        self.model_generator = self.__init_model__()

        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.input_question_key = config['input_question_key']
        self.input_sql_key = config['input_sql_key']
        self.input_schema_key = config['input_schema_key']
        self.num_threads = config['num_threads']
        self.max_retries = config['max_retries']
        self.exist_knowledge = config['exist_knowledge']


    def __init_model__(self) -> Union[LocalModelGenerator, APIGenerator_aisuite, APIGenerator_request]:
        generator_type = self.config["generator_type"].lower()
        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        raise ValueError(f"Invalid generator type: {generator_type}")


    def load_jsonl(self, file_path: str) -> List[Dict]:
        try:
            df = pd.read_json(file_path, lines=True, encoding='utf-8')
            return df.to_dict('records')
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise
    
    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        try:
            df = pd.DataFrame(data)
            df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        except Exception as e:
            logging.error(f"Error saving to {file_path}: {e}")
            raise

    def _generate_prompt(self, item: Dict) -> str:
        return self.prompt.extra_knowledge_prompt(item[self.input_question_key],
                                                  item[self.input_sql_key],
                                                  item[self.input_schema_key])
    
    def _parse_response(self, response: str) -> str:
        response = response.strip()
        upper_response = response.strip().upper()
        if not upper_response:
            return None
        
        if "RESULT: NO" in upper_response:
            return None
        
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.strip().startswith("RESULT:")
            )
            knowledge = result_line.split("RESULT:", 1)[1].strip()
            return knowledge if knowledge else None
        except (StopIteration, IndexError):
            logging.warning(f"Failed to parse response: {response[:200]}...")
            return None
    

    def _process_item_with_retry(self, item: Dict, retry_count: int = 2) -> Dict:
        try:
            prompt = self._generate_prompt(item)
            # logging.warning(prompt)
            response = self.model_generator.generate_text_from_input([prompt])
            # logging.warning(response)
            parsed_response = self._parse_response(response[0])
            # logging.warning(parsed_response)
            item['extra_knowledge'] = parsed_response
            return item
        except Exception as e:
            if retry_count < self.max_retries:
                # logging.warning(f"Retry {retry_count + 1} for item")
                return self._process_item_with_retry(item, retry_count + 1)
            # logging.error(f"Failed after {self.max_retries} retries for item : {e}")
            item['extra_knowledge'] = ''
            return item

    def run(self) -> None:
        # logging.info(f"Loading items from {self.input_path}")
        items = self.load_jsonl(self.input_path)
        if self.exist_knowledge:
            logging.info(f"Extra knowledge already exists, skipping generation.")
            self.save_jsonl(items, self.output_path)
        else:
            # logging.info(f"Start generating extra knowledge.")
            try:
                items = self.load_jsonl(self.input_path)
                
                question_id_to_index = {item['question_id']: idx for idx, item in enumerate(items)}
            
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = {
                        executor.submit(self._process_item_with_retry, item): item['question_id']
                        for item in tqdm(items, desc="Submitting tasks", unit="item")
                    }
                    
                    for future in tqdm(as_completed(futures), total=len(items),
                                desc="Generating Extra-Knowledge", unit="item"):
                        question_id = futures[future]
                        try:
                            processed_item = future.result()
                            items[question_id_to_index[question_id]] = processed_item
                        except Exception as e:
                            logging.error(f"Error processing question_id={question_id}: {e}")
                            continue
                
                self.save_jsonl(items, self.output_path)
                # logging.info(f"Successfully processed {len(items)} items to {self.output_path}")
                
            except Exception as e:
                logging.error(f"Fatal error in processing pipeline: {e}")
                raise