import pandas as pd
import re
from tqdm import tqdm
from typing import Dict, Optional
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class Text2SQLPromptGenerator(OperatorABC):
    def __init__(self, 
                database_manager: DatabaseManager,
                prompt_template: str = "",
                schema_config: Optional[Dict] = None
            ):

        if not prompt_template:
            self.prompt_template = '''Task Overview:
            /* Given the following database schema: */
            {schema}
            /* Answer the following: {question} */
            Let's think step by step'''
        else:
            self.prompt_template = prompt_template
        
        if not schema_config:
            self.schema_config = {
                'format': 'ddl',  # Optional: 'ddl', 'formatted_schema'
                'use_example': True  # Whether to include example data
            }
        else:
            self.schema_config = schema_config
        
        self.logger = get_logger()
        self.database_manager = database_manager
        self._validate_config()

    def _validate_config(self):
        if "{schema}" not in self.prompt_template or "{question}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {schema} and {question} placeholders")
        
        valid_formats = ['ddl', 'formatted_schema']
        if self.schema_config.get('format') not in valid_formats:
            raise ValueError(f"schema_config.format must be one of {valid_formats}")
        
        if not isinstance(self.schema_config.get('use_example'), bool):
            raise ValueError("schema_config.use_example must be a boolean")

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "从数据库提取Schema信息，结合自然语言问题生成提示词。其中提示词模版支持自定义。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题列名\n"
                "- input_db_id_key: 数据库ID列名\n"
                "- output_prompt_key: 输出prompt列名\n\n"
                "输出参数：\n"
                "- output_prompt_key: 生成的prompt"
            )
        elif lang == "en":
            return (
                "This operator generates prompts for Text2SQL tasks by extracting schema information from databases and combining it with natural language questions. The prompt template can be customized.\n\n"
                "Input parameters:\n"
                "- input_question_key: The name of the question column\n"
                "- input_db_id_key: The name of the database ID column\n"
                "- output_prompt_key: The name of the output prompt column\n\n"
                "Output parameters:\n"
                "- output_prompt_key: The generated prompt"
            )
        else:
            return "Prompt generator for Text2SQL tasks."

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

    def generate_prompt(self, db_id: str, question: str) -> str:
        formatted_schema = self.format_schema_according_to_config(db_id)
        generated_prompt = self.prompt_template.format(
            schema=formatted_schema, 
            question=question
        )
        return generated_prompt

    def _process_item(self, item: Dict) -> Dict:
        try:
            db_id = item[self.input_db_id_key]
            question = item[self.input_question_key]
            
            db_id = re.sub(r'[^A-Za-z0-9_]', '', str(db_id).replace('\n', ''))
            
            prompt = self.generate_prompt(db_id, question)
            
            result = {
                **item,
                self.output_sft_prompt_key: prompt
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing item: {e}")
            return {
                **item,
                self.output_sft_prompt_key: f"Error processing item: {e}",
                '_error': str(e)
            }

    def run(self, storage: DataFlowStorage, 
            input_question_key: str = "question",
            input_db_id_key: str = "db_id",
            output_prompt_key: str = "prompt"
        ):
        
        self.input_question_key = input_question_key
        self.input_db_id_key = input_db_id_key
        self.output_sft_prompt_key = output_prompt_key

        self.logger.info("Starting prompt generation...")
        raw_dataframe = storage.read("dataframe")
        
        required_cols = [input_question_key, input_db_id_key]
        missing_cols = [col for col in required_cols if col not in raw_dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        items = raw_dataframe.to_dict('records')
        final_results = []

        for item in tqdm(items, desc="Generating prompts"):
            result = self._process_item(item)
            final_results.append(result)
   
        if len(final_results) != len(items):
            self.logger.warning(f"Results count mismatch: expected {len(items)}, got {len(final_results)}")
        
        output_file = storage.write(pd.DataFrame(final_results))
        self.logger.info(f"Prompt generation completed, saved to {output_file}")

        return [self.output_sft_prompt_key]