from typing import Dict, Union
from tqdm import tqdm
import re
import time
from dataflow.prompts.text2sql import Text2SQLCorrespondenceFilterPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager

@prompt_restrict(Text2SQLCorrespondenceFilterPrompt)

@OPERATOR_REGISTRY.register()
class Text2SQLCorrespondenceFilter(OperatorABC):
    def __init__(self, 
            llm_serving: LLMServingABC, 
            database_manager: DatabaseManager,
            prompt_template: Union[Text2SQLCorrespondenceFilterPrompt, DIYPromptABC] = None
        ):
        self.llm_serving = llm_serving
        if prompt_template is None:
            prompt_template = Text2SQLCorrespondenceFilterPrompt()
        self.prompt_template = prompt_template
        self.database_manager = database_manager
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对条目进行过滤，检测SQL和自然语言问题是否对应。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_db_id_key: 输入数据库ID列名\n"
                "- input_question_key: 输入问题列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator filters items based on whether the SQL and question are corresponding.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_db_id_key: The name of the input database ID column\n"
                "- input_question_key: The name of the input question column\n\n"
            )
        else:
            return "SQL correspondence filter for Text2SQL tasks."

    def _parse_consistency_response(self, response):
        if not isinstance(response, str):
            self.logger.warning(f"Invalid response type: {type(response)}, expected str. Response: {response}")
            return False
        response_lower = response.lower() if response else ""
        pattern = r"```\s*(.*?)\s*```"
        ans_blocks = re.findall(pattern, response_lower, re.DOTALL)
        for ans_block in ans_blocks:
            if 'yes' in ans_block:
                return True
        return False
        
    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key, self.input_question_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_db_id_key: str = "db_id",
            input_question_key: str = "question",
            input_evidence_key: str = "evidence",
        ): 
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.input_question_key = input_question_key
        self.input_evidence_key = input_evidence_key
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        total_len = len(dataframe)
        
        prompts = []
        prompt_to_row_index = []  
        for row_index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing consistency check"):
            sql = row[self.input_sql_key]
            question = row.get(self.input_question_key)
            evidence = row.get(self.input_evidence_key, "")
            if question is None or str(question).strip() == "":
                continue
            if evidence:
                question = f"{question}\n{evidence}"
            db_id = row[self.input_db_id_key]
            
            create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(
                db_id
            )
            
            db_details = "\n\n".join(create_statements)
            prompt = self.prompt_template.build_prompt(question, sql, db_details)
            prompts.append(prompt)
            prompt_to_row_index.append(row_index)
        responses = self.llm_serving.generate_from_input(prompts, "")
        final_valid_indices = []
        for idx, response in enumerate(responses):
            conclusion = self._parse_consistency_response(response)
            if conclusion and idx < len(prompt_to_row_index):
                final_valid_indices.append(prompt_to_row_index[idx])

        correspondence_passed = len(final_valid_indices)
        self.logger.info(f"Correspondence check results: {correspondence_passed} passed, total {total_len}")
        
        if final_valid_indices:
            filtered_dataframe = dataframe.iloc[final_valid_indices].copy()
        else:
            self.logger.warning("No data passed all filters. Returning empty dataset.")
            filtered_dataframe = dataframe.iloc[0:0].copy()
        
        output_file = storage.write(filtered_dataframe)
        output_count = len(filtered_dataframe)
        
        self.logger.info(f"Filtered dataset saved to {output_file}")
        return []