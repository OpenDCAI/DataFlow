from typing import Dict
from tqdm import tqdm
import re
from dataflow.prompts.text2sql import TextSQLConsistencyPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class SQLConsistencyFilter(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, database_manager: DatabaseManager):
        self.llm_serving = llm_serving     
        self.prompt = TextSQLConsistencyPrompt()
        self.database_manager = database_manager
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对条目进行过滤，检测SQL和自然语言问题是否对应，即判断SQL是否能解决该问题。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_db_id_key: 输入数据库ID列名\n"
                "- input_question_key: 输入问题列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator filters items based on whether the SQL can solve the question.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_db_id_key: The name of the input database ID column\n"
                "- input_question_key: The name of the input question column\n\n"
            )
        else:
            return "SQL consistency filter for Text2SQL tasks."

    def _parse_consistency_response(self, response):
        response_lower = response.lower() if response else ""
        pattern = r"```\s*(.*?)\s*```"
        ans_blocks = re.findall(pattern, response_lower, re.DOTALL)
        for ans_block in ans_blocks:
            if 'yes' in ans_block:
                return True
        return False
        
    def generate_consistency_prompt(self, question, sql, schema):
        return self.prompt.text_sql_consistency_prompt(question, sql, schema)

    def format_schema_according_to_config(self, db_id: str) -> str:
        return self.database_manager.generate_ddl_without_examples(db_id)

    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key, self.input_question_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_db_id_key: str = "db_id",
            input_question_key: str = "question"
        ): 
        
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.input_question_key = input_question_key

        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        total_len = len(dataframe)
        prompts = []
        final_valid_indices = []
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing consistency check"):
            sql = row[self.input_sql_key]
            question = row[self.input_question_key]
            db_id = row[self.input_db_id_key]
            ddl = self.format_schema_according_to_config(db_id)
            prompt = self.prompt.text_sql_consistency_prompt(question, sql, ddl)
            prompts.append(prompt)
            
        responses = self.llm_serving.generate_from_input(prompts, "")
        for idx, response in enumerate(responses):
            conclusion = self._parse_consistency_response(response)
            if conclusion:
                final_valid_indices.append(idx)

        consistency_passed = len(final_valid_indices)
        self.logger.info(f"Consistency check results: {consistency_passed} passed, total {total_len}")
        
        if final_valid_indices:
            filtered_dataframe = dataframe.loc[final_valid_indices].copy()
        else:
            self.logger.warning("No data passed all filters. Returning empty dataset.")
            filtered_dataframe = dataframe.iloc[0:0].copy()
        
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtered dataset saved to {output_file}")
        return []