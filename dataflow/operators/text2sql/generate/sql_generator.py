import os
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import re
from dataflow.prompts.text2sql import SQLGenerationPrompt
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class SQLGenerator(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC, 
                 database_manager: DatabaseManager,
                 generate_num: int = 300):
        self.llm_serving = llm_serving
        self.logger = get_logger()
        self.database_manager = database_manager
        self.generate_num = generate_num
        self.prompt = SQLGenerationPrompt()
        self.complexity2criterion = {
            "Simple": self.prompt.simple_criterion,
            "Moderate": self.prompt.moderate_criterion,
            "Complex": self.prompt.complex_criterion, 
            "Highly Complex": self.prompt.highly_complex_criterion
        }
        random.seed(42)

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

    def obtain_db_schema(self, db_manager: DatabaseManager, db_id: str) -> tuple:
        return db_manager.get_table_names_and_create_statements(db_id)

    def obtain_insert_statements(self, db_manager: DatabaseManager, db_id: str, table_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        return db_manager.get_insert_statements(db_id, table_names, limit=2)

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
            output_sql_key: str = "sql",
            output_db_id_key: str = "db_id"
        ):
        self.output_sql_key = output_sql_key
        self.output_db_id_key = output_db_id_key
        raw_dataframe = storage.read("dataframe")
        functions = self.prompt.sqlite_funcs()
        
        db_names = self.database_manager.list_databases()
        prompts = []
        self.logger.info(f"Generating {self.generate_num} SQLs for each database")

        
        for db_name in tqdm(db_names, desc="Processing Databases"):
            table_names, create_statements = self.obtain_db_schema(self.database_manager, db_name)

            if not table_names:
                self.logger.warning(f"Database {db_name} has no tables")
                continue

            table_name2insert_statements = self.obtain_insert_statements(self.database_manager, db_name, table_names)

            for _ in range(self.generate_num):
                complexity = random.choice(["Simple", "Moderate", "Complex", "Highly Complex"])

                insert_statements = []
                for table_name in table_names:
                    insert_statements.extend(table_name2insert_statements.get(table_name, []))

                if len(insert_statements) == 0:
                    db_value_prompt = ""
                else:
                    if len(insert_statements) > 4:
                        insert_statements = random.sample(insert_statements, 4)
                    db_value_prompt = self.prompt.insert_stmts_template(
                        insert_statements="\n\n".join(insert_statements)
                    )

                function_num = random.randint(0, 2)
                if function_num == 0:
                    sql_function_prompt = "### SQL Functions\nYou can use any function supported by the database engine."
                else:
                    sql_funcs = ""
                    sampled_functions = random.sample(functions, min(function_num, len(functions)))
                    for idx, func in enumerate(sampled_functions):
                        sql_funcs += f"Function {idx + 1}:\n{func.strip()}\n"
                    sql_function_prompt = self.prompt.sql_func_template(sql_funcs=sql_funcs)

                column_count = np.random.geometric(0.6, 1)[0]

                prompt = self.prompt.sql_synthesis_prompt(
                    schema_str="\n\n".join(create_statements),
                    sql_function_prompt=sql_function_prompt.strip(),
                    db_value_prompt=db_value_prompt.strip(),
                    complexity=complexity,
                    criterion=self.complexity2criterion[complexity].strip(),
                    db_engine=self.database_manager.db_type,
                    column_count=column_count
                )

                prompts.append({"prompt": prompt, "db_id": db_name, "complexity": complexity})
            
        if not prompts:
            self.logger.warning("No prompts generated, please check the database path and file")
            return [self.output_sql_key, self.output_db_id_key]
            
        db_ids = [data["db_id"] for data in prompts]
        prompt_list = [data["prompt"] for data in prompts]
        
        try:
            responses = self.llm_serving.generate_from_input(prompt_list, "")
        except Exception as e:
            self.logger.error(f"Failed to generate SQLs: {e}")
            responses = [""] * len(prompt_list)
            
        results = [
            {
                self.output_db_id_key: db_id,
                self.output_sql_key: self.parse_response(response)
            }
            for db_id, response in zip(db_ids, responses)
        ]
        
        output_file = storage.write(pd.DataFrame(results))
        return [self.output_sql_key, self.output_db_id_key]