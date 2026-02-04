import random
import pandas as pd
import re
from dataflow.prompts.text2sql import SelectSQLGeneratorPrompt
from dataflow.prompts.text2sql import SelectVecSQLGeneratorPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager
from typing import Union

@prompt_restrict(SelectSQLGeneratorPrompt, SelectVecSQLGeneratorPrompt)

@OPERATOR_REGISTRY.register()
class SQLGenerator(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC, 
                 database_manager: DatabaseManager,
                 generate_num: int = 300,
                 prompt_template: Union[SelectSQLGeneratorPrompt, SelectVecSQLGeneratorPrompt, DIYPromptABC] = None
        ):
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.database_manager = database_manager
        self.generate_num = generate_num
        if prompt_template is None:
            self.prompt_template = SelectSQLGeneratorPrompt()
        else:
            self.prompt_template = prompt_template

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "基于数据库信息，合成SQL，覆盖不同的难度、数据库Schema、函数和风格。\n\n"
                "输出参数：\n"
                "- output_sql_key: 输出SQL列名\n"
                "- output_db_id_key: 数据库ID列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator synthesizes SQL based on database information, covering different complexities, schemas, functions, and styles.\n\n"
                "Output parameters:\n"
                "- output_sql_key: The name of the output SQL column\n"
                "- output_db_id_key: The name of the database ID column\n\n"
            )
        else:
            return "SQL generator for Text2SQL tasks."

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

    def run(self, storage: DataFlowStorage,
            output_sql_key: str = "SQL",
            output_db_id_key: str = "db_id",
            output_sql_complexity_key: str = "sql_complexity_type"
        ):
        self.output_sql_key = output_sql_key
        self.output_db_id_key = output_db_id_key
        self.output_sql_complexity_key = output_sql_complexity_key
        raw_dataframe = storage.read("dataframe")
        
        db_names = self.database_manager.list_databases()
        prompts = []
        self.logger.info(f"Generating {self.generate_num} SQLs for each database")

        for db_name in tqdm(db_names, desc="Processing Databases"):
            sum_generate_num = self.generate_num
            create_statements, insert_statements = self.database_manager.get_create_statements_and_insert_statements(db_name)

            for _ in range(sum_generate_num):
                prompt, sql_complexity_type = self.prompt_template.build_prompt(
                    insert_statements=insert_statements,
                    create_statements=create_statements,
                    db_engine=self.database_manager.db_type
                )
                prompts.append({"prompt": prompt, "db_id": db_name, "sql_complexity_type": sql_complexity_type})
            
        if not prompts:
            self.logger.warning("No prompts generated, please check the database path and file")
            return [self.output_sql_key, self.output_db_id_key, self.output_sql_complexity_key]
            
        db_ids = [data["db_id"] for data in prompts]
        prompt_list = [data["prompt"] for data in prompts]
        sql_complexity_types = [data["sql_complexity_type"] for data in prompts]

        try:
            responses = self.llm_serving.generate_from_input(prompt_list, "")
        except Exception as e:
            self.logger.error(f"Failed to generate SQLs: {e}")
            responses = [""] * len(prompt_list)
            
        results = [
            {
                self.output_db_id_key: db_id,
                self.output_sql_key: self.parse_response(response),
                self.output_sql_complexity_key: sql_complexity_type
            }
            for db_id, response, sql_complexity_type in zip(db_ids, responses, sql_complexity_types)
        ]
        
        output_file = storage.write(pd.DataFrame(results))
        return [self.output_sql_key, self.output_db_id_key, self.output_sql_complexity_key]
