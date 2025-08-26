import random
from typing import Dict, List, Optional
import pandas as pd
import re
from dataflow.prompts.text2sql import SQLVariationGeneratorPrompt
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import (DataFlowStorage, RESERVED_SYS_FIELD_LIST, RESERVED_USER_FIELD_LIST,
                                    SYS_FIELD_PREFIX, USER_FIELD_PREFIX)
from dataflow.utils.text2sql.database_manager import DatabaseManager
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC


@OPERATOR_REGISTRY.register()
class SQLVariationGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, 
                 database_manager: DatabaseManager,
                 num_variations: int = 10,
                 prompt_template = None
                 ):
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.database_manager = database_manager
        if prompt_template is None:
            self.prompt_template = SQLVariationGeneratorPrompt()
        else:
            self.prompt_template = prompt_template
        self.num_variations = num_variations
        random.seed(42)

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

    def get_create_statements_and_insert_statements(self, db_id: str) -> str:
        return self.database_manager.get_create_statements_and_insert_statements(db_id)

    def parse_response(self, response):
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
            input_sql_key: str = "sql",
            input_db_id_key: str = "db_id"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        original_count = len(dataframe)
        prompts_and_metadata = []
        original_row_indices = []
        
        for row_idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Generating SQL Variations"):
            try:
                create_statements, insert_statements = self.get_create_statements_and_insert_statements(row[self.input_db_id_key])
                table_names = self.database_manager.get_table_names(row[self.input_db_id_key])
                
                original_sql = row[self.input_sql_key]

                for _ in range(self.num_variations):
                    insert_statements = []
                    for table_name in table_names:
                        insert_statements.extend(insert_statements.get(table_name, []))
                    
                    prompt = self.prompt_template.build_prompt(
                        original_sql=original_sql,
                        schema_str="\n\n".join(create_statements),
                        db_value_prompt=insert_statements,
                        db_engine=self.database_manager.db_type
                    )
                    
                    prompts_and_metadata.append((
                        prompt, 
                        row[self.input_db_id_key]
                    ))
                    original_row_indices.append(row_idx)
                    
            except Exception as e:
                self.logger.error(f"Error processing database {row[self.input_db_id_key]}: {e}")
                continue  
        
        if prompts_and_metadata:
            try:
                prompts = [prompt for prompt, db_id in prompts_and_metadata]
                responses = self.llm_serving.generate_from_input(prompts, system_prompt="")
                for i, ((prompt, db_id), response) in enumerate(zip(prompts_and_metadata, responses)):
                    sql = self.parse_response(response)
                    if sql:
                        original_row_idx = original_row_indices[i]
                        original_row = dataframe.iloc[original_row_idx]

                        # 新建全 None 的新行
                        new_row = {col: None for col in dataframe.columns}

                        # 设置 db_id 和 sql
                        new_row[self.input_db_id_key] = db_id
                        new_row[self.input_sql_key] = sql

                        # 处理保留字段
                        for sys_field in RESERVED_SYS_FIELD_LIST:
                            sys_col = f"{SYS_FIELD_PREFIX}{sys_field}"
                            if sys_col in dataframe.columns and sys_col in original_row:
                                new_row[sys_col] = original_row[sys_col]
                        for user_field in RESERVED_USER_FIELD_LIST:
                            user_col = f"{USER_FIELD_PREFIX}{user_field}"
                            if user_col in dataframe.columns and user_col in original_row:
                                new_row[user_col] = original_row[user_col]

                        # 将新行添加到dataframe中
                        dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error generating SQL variations: {e}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated {len(dataframe)} records (original: {original_count}, variations: {len(dataframe) - original_count})")
        return []