from typing import Dict, Optional, Tuple, List
import pandas as pd
import re
from dataflow.prompts.text2sql import CotGenerationPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class Text2SQLCoTGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, 
                database_manager: DatabaseManager,
                schema_config: Optional[Dict] = None,
                max_retries: int = 3,
                enable_retry: bool = True,
                timeout: int = 5
                 ):
        self.llm_serving = llm_serving
        self.database_manager = database_manager
        self.cot_generation_prompt = CotGenerationPrompt()
        self.logger = get_logger()
        if not schema_config:
            self.schema_config = {
                'format': 'ddl',  # Optional: 'ddl', 'formatted_schema'
                'use_example': True  # Whether to include example data
            }
        else:
            self.schema_config = schema_config
        self.max_retries = max_retries
        self.enable_retry = enable_retry
        self.timeout = timeout
        self._validate_config()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对于每个条目，生成从自然语言问题和数据库Schema到SQL的CoT长链路推理过程。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_question_key: 输入问题列名\n"
                "- input_db_id_key: 输入数据库ID列名\n\n"
                "输出参数：\n"
                "- output_cot_key: 输出CoT列名"
            )
        elif lang == "en":
            return (
                "This operator generates CoT for SQL with long chain reasoning from natural language questions and database schemas.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_question_key: The name of the input question column\n"
                "- input_db_id_key: The name of the input database ID column\n\n"
                "Output parameters:\n"
                "- output_cot_key: The name of the output CoT column"
            )
        else:
            return "CoT generator for Text2SQL tasks with long chain reasoning."

    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key, self.input_question_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _validate_config(self):
        valid_formats = ['ddl', 'formatted_schema']
        if self.schema_config.get('format') not in valid_formats:
            raise ValueError(f"schema_config.format must be one of {valid_formats}")
        
        if not isinstance(self.schema_config.get('use_example'), bool):
            raise ValueError("schema_config.use_example must be a boolean")

    def get_schema_for_db(self, db_id: str) -> Dict:
        return self.database_manager.get_database_schema(db_id)

    def format_schema_according_to_config(self, db_id: str) -> str:
        format_type = self.schema_config.get('format', 'formatted_schema')
        use_example = self.schema_config.get('use_example', True)
        
        if format_type == 'ddl':
            if use_example:
                return self.database_manager.generate_ddl_with_examples(db_id)
            else:
                return self.database_manager.generate_ddl_without_examples(db_id)
        elif format_type == 'formatted_schema':
            if use_example:
                return self.database_manager.generate_formatted_schema_with_examples(db_id)
            else:
                return self.database_manager.generate_formatted_schema_without_examples(db_id)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def generate_cot_synthesis_prompts(self, schema: str, question: str, sql: str) -> str:
        return self.cot_generation_prompt.text2sql_cot_prompt(
                schema,
                question,
                sql
            )
    
    def extract_sql(self, response):
        pattern = r"```sql\s*(.*?)\s*```"
        sql_blocks = re.findall(pattern, response, re.DOTALL)
        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            return ""
    
    def _parse_response(self, response: str, gold_sql: str, db_id: str) -> Tuple[Optional[str], bool]:
        generated_sql = self.extract_sql(response)
        if not generated_sql:
            return None, False
        try:
            ans = self.database_manager.compare_sql(db_id, generated_sql, gold_sql, 5.0)
            
            if ans:
                return generated_sql, True
            return generated_sql, False
                    
        except Exception as e:
            self.logger.error(f"SQL execution failed: {db_id}, Error: {e}")
            return generated_sql, False

    def _process_items_with_retry(self, items: List[Dict], max_retries: int = 3) -> List[Dict]:
        results = []
        failed_items = items.copy()
        
        for retry_round in range(max_retries):
            if not failed_items:
                break
                
            self.logger.info(f"Start {retry_round + 1} round processing, {len(failed_items)} items to process")
            
            prompts = []
            for item in failed_items:
                db_id = item.get(self.input_db_id_key)
                formatted_schema = self.format_schema_according_to_config(db_id)
                question = item.get(self.input_question_key)
                sql = item.get(self.input_sql_key)
                cot_prompt = self.generate_cot_synthesis_prompts(formatted_schema, question, sql)
                prompts.append(cot_prompt)
            
            cot_responses = self.llm_serving.generate_from_input(prompts, "")
            
            comparisons = []
            valid_items_with_responses = []
            
            for item, response in zip(failed_items, cot_responses):
                db_id = item.get(self.input_db_id_key)
                gold_sql = item.get(self.input_sql_key)
                generated_sql = self.extract_sql(response)
                
                if generated_sql:
                    comparisons.append({
                        'db_id': db_id,
                        'sql1': generated_sql,
                        'sql2': gold_sql,
                        'timeout': self.timeout,
                        'operation_id': f"{db_id}_{len(valid_items_with_responses)}"
                    })
                    valid_items_with_responses.append((item, response, generated_sql))
            
            if comparisons:
                try:
                    batch_results = self.database_manager.batch_compare_sql(comparisons)
                    
                    current_round_failed = []
                    for (item, response, generated_sql), batch_result in zip(valid_items_with_responses, batch_results):
                        db_id = item.get(self.input_db_id_key)
                        
                        if batch_result.success and batch_result.data:
                            results.append({
                                **item,
                                self.output_cot_key: response
                            })
                            self.logger.debug(f"Successfully processed {db_id} (Round {retry_round + 1})")
                        else:
                            current_round_failed.append(item)
                            if batch_result.error:
                                self.logger.debug(f"SQL comparison failed for {db_id}: {batch_result.error}")
                    
                    for item, response in zip(failed_items, cot_responses):
                        if item not in [valid_item for valid_item, _, _ in valid_items_with_responses]:
                            current_round_failed.append(item)
                            
                except Exception as e:
                    self.logger.error(f"Batch SQL comparison failed: {e}")
                    current_round_failed = []
                    for item, response in zip(failed_items, cot_responses):
                        db_id = item.get(self.input_db_id_key)
                        gold_sql = item.get(self.input_sql_key)
                        parsed_response, success = self._parse_response(response, gold_sql, db_id)
                        
                        if success and parsed_response:
                            results.append({
                                **item,
                                self.output_cot_key: response
                            })
                            self.logger.debug(f"Successfully processed {db_id} (Round {retry_round + 1})")
                        else:
                            current_round_failed.append(item)
            else:
                current_round_failed = failed_items
            
            failed_items = current_round_failed
            self.logger.info(f"Text2SQL CoT Generation: Round {retry_round + 1} completed, Success: {len(results)}, Failed: {len(failed_items)}")
        
        if failed_items:
            self.logger.warning(f"Still {len(failed_items)} items failed, will be discarded")
            for item in failed_items:
                self.logger.debug(f"Discarded failed item: {item.get(self.input_db_id_key)}")
        
        return results

    def run(self, storage: DataFlowStorage, 
            input_sql_key: str = "SQL",
            input_question_key: str = "question",
            input_db_id_key: str = "db_id",
            output_cot_key: str = "cot_reasoning"
        ):
        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.output_cot_key = output_cot_key
        
        self.logger.info("Starting CoT generation...")
        raw_dataframe = storage.read("dataframe")
        self.check_column(raw_dataframe)
        items = raw_dataframe.to_dict('records')
        results = self._process_items_with_retry(items, self.max_retries)
        
        if not results:
            self.logger.warning("No CoT results generated")
            return []
        
        output_df = pd.DataFrame(results)
        output_file = storage.write(output_df)
        self.logger.info(f"CoT generation completed, saved to {output_file}")
        self.logger.info(f"Processed {len(results)} items, original {len(items)} items")
        
        return [self.output_cot_key]