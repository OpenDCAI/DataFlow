import json
import logging
from typing import Dict, List
from dataflow.generator.utils.Prompts import FinalPromptGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class PromptGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.prompt = FinalPromptGeneration()
        self._lock = threading.Lock()

        self.input_file = config['input_file']
        self.output_file = config['output_file']
        
        self.input_question_key = config.get('input_question_key', 'question')
        self.input_sql_key = config.get('input_sql_key', 'sql')
        self.input_evidence_key = config.get('input_evidence_key', 'evidence')
        self.input_schema_key = config.get('input_schema_key', 'schema')
        self.prompt_type = config['prompt_type']
        self.use_cot = config['use_cot']
        self.num_threads = config.get('num_threads', 5)
        self.timeout = config.get('timeout', 60)  

    def load_jsonl(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise

    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        try:
            with self._lock, open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        except IOError as e:
            logging.error(f"Error saving to {file_path}: {e}")
            raise

    def _process_item(self, item: Dict) -> Dict:
        # logging.warning(f"Processing item: {item}")
        if self.prompt_type == 'dail-sql':
            if self.use_cot:
                item['final_prompt'] = self.prompt.dial_sql_cot_prompt(
                        question=item.get(self.input_question_key),
                        sql=item.get(self.input_sql_key),
                        schema=item.get(self.input_schema_key),
                        evidence=item.get(self.input_evidence_key)
                    )
            else:
                item['final_prompt'] = self.prompt.dial_sql_non_cot_prompt(
                        question=item.get(self.input_question_key),
                        sql=item.get(self.input_sql_key),
                        schema=item.get(self.input_schema_key),
                        evidence=item.get(self.input_evidence_key)
                    )
        elif self.prompt_type == 'omni-sql':
            if self.use_cot:
                    item['final_prompt'] = self.prompt.omni_sql_cot_prompt(
                        question=item.get(self.input_question_key),
                        sql=item.get(self.input_sql_key),
                        schema=item.get(self.input_schema_key),
                        evidence=item.get(self.input_evidence_key)
                    )
            else:
                item['final_prompt'] = self.prompt.omni_sql_non_cot_prompt(
                        question=item.get(self.input_question_key),
                        sql=item.get(self.input_sql_key),
                        schema=item.get(self.input_schema_key),
                        evidence=item.get(self.input_evidence_key)
                    )
        return item

    def run(self) -> None:
        # logging.info(f"Starting processing on {self.input_path}")
        try:
            items = self.load_jsonl(self.input_path)
            question_id_to_index = {item['question_id']: idx for idx, item in enumerate(items)}
            # logging.warning(item_map)

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._process_item, item): item['question_id']
                    for item in items
                }
                    
                for future in tqdm(as_completed(futures, timeout=self.timeout), 
                             total=len(futures),
                             desc="Generating prompts",
                             unit="item"):
                    question_id = futures[future]
                    try:
                        processed_item = future.result()
                        items[question_id_to_index[question_id]].update(processed_item)
                    except Exception as e:
                        logging.error(f"Error processing question_id={question_id}: {e}")
                        continue
            
            self.save_jsonl(items, self.output_path)
            # logging.info(f"Successfully processed {len(items)} items to {self.output_path}")
            
        except Exception as e:
            logging.error(f"Fatal error in processing pipeline: {e}")
            raise