import random
import pandas as pd
import re
from dataflow.prompts.text2sql import SQLVariationGeneratorPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import (DataFlowStorage, RESERVED_SYS_FIELD_LIST, RESERVED_USER_FIELD_LIST,
                                    SYS_FIELD_PREFIX, USER_FIELD_PREFIX)
from dataflow.utils.text2sql.database_manager import DatabaseManager
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from typing import Union

@prompt_restrict(SQLVariationGeneratorPrompt)

@OPERATOR_REGISTRY.register()
class SQLVariationGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, 
                 database_manager: DatabaseManager,
                 num_variations: int = 10,
                 prompt_template: Union[SQLVariationGeneratorPrompt, DIYPromptABC] = None
                 ):
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.database_manager = database_manager
        if prompt_template is None:
            self.prompt_template = SQLVariationGeneratorPrompt()
        else:
            self.prompt_template = prompt_template
        self.num_variations = num_variations

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对于每个条目，基于已有的SQL，指导模型生成SQL的变种，即在原有SQL的基础上，进行数据替换、函数变换、难度变换等操作，生成更加丰富的SQL。\n\n"
                "输入参数：\n"
                "- input_sql_key: SQL列名\n"
                "- input_db_id_key: 数据库ID列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator generates variations of SQL based on existing SQLs, including data replacement, function transformation, and difficulty transformation, to generate more diverse SQLs.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the SQL column\n"
                "- input_db_id_key: The name of the database ID column\n\n"
            )
        else:
            return "SQL variation generator for Text2SQL tasks."

    def parse_response(self, response):
        if not isinstance(response, str):
            self.logger.warning(f"Invalid response type: {type(response)}, expected str. Response: {response}")
            return ""
        if not response:
            return ""
                
        pattern = r"```sql\s*(.*?)\s*```"
        sql_blocks = re.findall(pattern, response, re.DOTALL)
            
        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            self.logger.warning("No SQL code block found in the response")
            return ""
    
    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_db_id_key: str = "db_id",
            output_sql_variation_type_key: str = "sql_variation_type"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        self.output_sql_variation_type_key = output_sql_variation_type_key

        prompts_and_metadata = []
        # Phase 1: Prompt building
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Generating SQL Variations"):
            try:
                create_statements, insert_statements = self.database_manager.get_create_statements_and_insert_statements(row[self.input_db_id_key])
                original_sql = row[self.input_sql_key]

                for _ in range(self.num_variations):
                    prompt, variation_type = self.prompt_template.build_prompt(
                        original_sql=original_sql,
                        create_statements=create_statements,
                        insert_statements=insert_statements,
                        db_engine=self.database_manager.db_type
                    )

                    prompts_and_metadata.append((
                        prompt, 
                        row[self.input_db_id_key],
                        variation_type,
                        row.to_dict()
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error processing database {row[self.input_db_id_key]}: {e}")
                continue
        
        new_rows = []
        if prompts_and_metadata:
            try:
                prompts = [x[0] for x in prompts_and_metadata]
                responses = self.llm_serving.generate_from_input(prompts, system_prompt="")
                
                # Phase 2: Post-processing
                for i, ((prompt, db_id, variation_type, original_row), response) in enumerate(zip(prompts_and_metadata, responses)):
                    sql = self.parse_response(response)
                    if sql:
                        new_row = {col: None for col in dataframe.columns}

                        new_row[self.input_db_id_key] = db_id
                        new_row[self.input_sql_key] = sql
                        new_row[output_sql_variation_type_key] = variation_type

                        new_rows.append(new_row)
                
            except Exception as e:
                self.logger.error(f"Error generating SQL variations: {e}")

        if new_rows:
            dataframe = pd.DataFrame(new_rows)

        output_file = storage.write(dataframe)
        output_count = len(dataframe)
        
        self.logger.info(f"Generated {output_count} records")
        return [self.input_sql_key, self.input_db_id_key, self.output_sql_variation_type_key]