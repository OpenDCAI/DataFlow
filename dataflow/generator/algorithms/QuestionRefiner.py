from typing import Dict, List, Optional, Union
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataflow.generator.utils.Prompts import QuestionRefinePrompt
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite

class QuestionRefiner:
    def __init__(self, args: Dict):
        self.config = args
        
        self.prompt = QuestionRefinePrompt()
        self.model_generator = self.__init_model__()
        self._lock = threading.Lock() 

        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.output_refined_question_key = config.get('output_refined_question_key')
        self.input_db_key = config.get('input_db_key', 'id')
        self.num_threads = config.get('num_threads', 5)
        self.max_retries = config.get('max_retries', 3)
        self.input_question_key = config.get('input_question_key', 'question')



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
        file_path = self.input_file
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON in line: {line.strip()}. Error: {e}")
        except IOError as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        file_path = self.output_file
        try:
            with self._lock: 
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
        except IOError as e:
            logging.error(f"Error saving to {file_path}: {e}")
            raise

    def _generate_prompt(self, item: Dict) -> str:
        return self.prompt.question_refine_prompt(item['question'])
    
    def _parse_response(self, response: str, original_question: str) -> str:
        if not response:
            return original_question
            
        response_upper = response.upper()
        if "RESULT: NO" in response_upper:
            return original_question
            
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.upper().startswith("RESULT:")
            )
            return result_line.split("RESULT:", 1)[1].strip()
        except (StopIteration, IndexError):
            logging.warning(f"Unexpected response format: {response[:200]}...")
            return original_question

    def _process_item_with_retry(self, item: Dict, retry_count: int = 2) -> Dict:
        try:
            prompt = self._generate_prompt(item)
            response = self.model_generator.generate_text_from_input([prompt])
            parsed_response = self._parse_response(response[0], item['question'])
            # logging.warning(parsed_response)
            # print(parsed_response)
            item[self.output_refined_question_key] = parsed_response
            return item
        except Exception as e:
            if retry_count < self.max_retries:
                # logging.warning(f"Retry {retry_count + 1} for item")
                return self._process_item_with_retry(item, retry_count + 1)
            # logging.error(f"Failed after {self.max_retries} retries for item: {e}")
            item[self.output_refined_question_key] = item['question'] 
            return item

    def run(self) -> None:
        # logging.info(f"Loading items from {self.input_file}")
        try:
            items = self.load_jsonl(self.input_file)

            question_id_to_index = {item['question_id']: idx for idx, item in enumerate(items)}
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._process_item_with_retry, item): item['question_id']
                    for item in tqdm(items, desc="Submitting tasks", unit="item")
                }
                    
                for future in tqdm(as_completed(futures), total=len(items), 
                             desc="Refining Questions", unit="item"):
                    question_id = futures[future]
                    try:
                        processed_item = future.result()
                        items[question_id_to_index[question_id]] = processed_item
                    except Exception as e:
                        logging.error(f"Error processing question_id={question_id}: {e}")
                        continue
            
            self.save_jsonl(items, self.output_file)
            # logging.info(f"Successfully processed {len(items)} items to {self.output_file}")
            
        except Exception as e:
            logging.error(f"Fatal error in processing pipeline: {e}")
            raise