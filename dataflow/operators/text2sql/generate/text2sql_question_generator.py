import json
import random
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

from dataflow.prompts.text2sql import QuestionGenerationPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class Text2SQLQuestionGenerator(OperatorABC):
    def __init__(self, 
                llm_serving: LLMServingABC, 
                embedding_serving: LLMServingABC, 
                database_manager: DatabaseManager, 
                question_candidates_num: int = 5):

        self.llm_serving = llm_serving
        self.embedding_serving = embedding_serving
        self.database_manager = database_manager
        self.prompt = QuestionGenerationPrompt()
        self.logger = get_logger()
        self.question_candidates_num = question_candidates_num
        random.seed(42)

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对于每个条目，如果自然语言问题为空，生成SQL对应的自然语言问题。为保证正确，生成多个候选问题，并选择最优的。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_db_id_key: 数据库ID列名\n\n"
                "输出参数：\n"
                "- output_question_key: 输出问题列名"
            )
        elif lang == "en":
            return (
                "This operator generates natural language questions for Text2SQL tasks if the natural language question is empty. Multiple candidate questions are generated to ensure correctness.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_db_id_key: The name of the database ID column\n\n"
                "Output parameters:\n"
                "- output_question_key: The name of the output question column"
            )
        else:
            return "Question generator for Text2SQL tasks."

    def extract_column_descriptions(self, create_statements):
        column_name2column_desc = dict()
        pattern = r'"(\w+)"\s+\w+\s*/\*\s*(.*?)\s*\*/'

        for create_statement in create_statements:
            matches = re.findall(pattern, create_statement)

            for column_name, description in matches:
                column_name = column_name.lower()
                if column_name not in column_name2column_desc:
                    column_name2column_desc[column_name] = description

        return column_name2column_desc

    def parse_llm_response(self, response, style):
        explanation_pattern = re.compile(r'\[EXPLANATION-START\](.*?)\[EXPLANATION-END\]', re.DOTALL)
        question_pattern = re.compile(r'\[QUESTION-START\](.*?)\[QUESTION-END\]', re.DOTALL)
        external_knowledge_pattern = re.compile(r'\[EXTERNAL-KNOWLEDGE-START\](.*?)\[EXTERNAL-KNOWLEDGE-END\]', re.DOTALL)

        explanation_match = explanation_pattern.search(response)
        question_match = question_pattern.search(response)
        external_knowledge_match = external_knowledge_pattern.search(response)

        explanation_content = explanation_match.group(1).strip() if explanation_match else ""
        question_content = question_match.group(1).strip() if question_match else ""
        external_knowledge_content = external_knowledge_match.group(1).strip() if external_knowledge_match else ""

        if explanation_content == "" or question_content == "":
            return None
        else:
            return {
                "question": question_content.strip(),
                "external_knowledge": external_knowledge_content.strip()
            }

    def select_best_question(self, question_candidates, start_idx, embeddings):
        if len(question_candidates) == 0:
            return None
        elif len(question_candidates) == 1:
            return question_candidates[0]
        elif len(question_candidates) == 2:
            return random.sample(question_candidates, 1)[0]
        else:
            end_idx = start_idx + len(question_candidates)
            candidate_embeddings = embeddings[start_idx:end_idx]
            distance_matrix = cdist(candidate_embeddings, candidate_embeddings, metric='cosine')
            distance_sums = distance_matrix.sum(axis=1)
            min_index = np.argmin(distance_sums)
            return question_candidates[min_index]

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "sql",
            input_db_id_key: str = "db_id",
            output_question_key: str = "question"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.output_question_key = output_question_key
        
        raw_dataframe = storage.read("dataframe")
        
        existing_data = []
        raw_data = []
        
        if self.output_question_key in raw_dataframe.columns:
            for _, row in raw_dataframe.iterrows():
                if pd.notna(row.get(self.output_question_key)) and row.get(self.output_question_key) is not None:
                    existing_data.append(row.to_dict())
                else:
                    raw_data.append(row.to_dict())
        else:
            raw_data = [row.to_dict() for _, row in raw_dataframe.iterrows()]
        
        styles = ["Formal", "Colloquial", "Imperative", "Interrogative", "Descriptive", "Concise", "Vague", "Metaphorical"]
        db_ids = list(set([data[self.input_db_id_key] for data in raw_data]))
        db_id2column_info = dict()
        style2desc = self.prompt.get_style2desc()
        
        for db_id in tqdm(db_ids, desc="Extracting database schema"):
            _, create_statements = self.database_manager.get_table_names_and_create_statements(db_id)
            db_id2column_info[db_id] = self.extract_column_descriptions(create_statements)
        
        self.logger.info("Generating question candidates...")
        prompts = []
        prompt_data_mapping = []
        
        for data in tqdm(raw_data, desc="Preparing prompts"):
            style_name = random.sample(styles, 1)[0]
            column_name2column_desc = db_id2column_info[data[self.input_db_id_key]]
            used_column_name2column_desc = dict()
            
            for column_name, column_desc in column_name2column_desc.items():
                if column_name.lower() in data[self.input_sql_key].lower():
                    used_column_name2column_desc[column_name] = column_desc

            if style_name in ["Vague", "Metaphorical"]:
                steps = self.prompt.get_steps_w_ek()
                guidelines = self.prompt.get_guidelines_w_ek()
                instruction = self.prompt.get_instruction_w_ek()
                output_format = self.prompt.get_output_format_w_ek()
            else:
                steps = self.prompt.get_steps_wo_ek()
                guidelines = self.prompt.get_guidelines_wo_ek()
                instruction = self.prompt.get_instruction_wo_ek()
                output_format = self.prompt.get_output_format_wo_ek()

            prompt = self.prompt.question_synthesis_prompt(
                style_desc=style2desc[style_name].strip(),
                engine=self.database_manager.db_type,
                column_info=json.dumps(used_column_name2column_desc, indent=2, ensure_ascii=False).strip(),
                sql=data[self.input_sql_key].strip(),
                steps=steps.strip(),
                guidelines=guidelines.strip(),
                output_format=output_format.strip(),
                instruction=instruction.strip()
            )
            
            for _ in range(self.question_candidates_num):
                prompts.append(prompt)
                prompt_data_mapping.append({**data, "style": style_name})

        responses = self.llm_serving.generate_from_input(prompts, system_prompt="You are a helpful assistant.")
        
        self.logger.info("Parsing responses and organizing candidates...")
        grouped_responses = [responses[i:i+self.question_candidates_num] for i in range(0, len(responses), self.question_candidates_num)]

        all_question_candidates = []
        question_groups = [] 
        embedding_texts = []
        
        for data, response_group in zip(raw_data, grouped_responses):
            question_candidates = []
            for response in response_group:
                parsed_response = self.parse_llm_response(response, data.get("style", "Formal"))
                if parsed_response:
                    question_candidates.append(parsed_response)
                    text = parsed_response["external_knowledge"] + " " + parsed_response["question"]
                    embedding_texts.append(text.strip())
            
            question_groups.append(question_candidates)
            all_question_candidates.extend(question_candidates)
        
        self.logger.info("Generating embeddings for all question candidates...")
        if embedding_texts:
            embeddings = self.embedding_serving.generate_embedding_from_input(embedding_texts)
        else:
            embeddings = []
        
        processed_results = []
        failed_data = []
        embedding_start_idx = 0
        
        for data, question_candidates in zip(raw_data, question_groups):
            if question_candidates:
                best_question = self.select_best_question(
                    question_candidates, 
                    embedding_start_idx, 
                    embeddings
                )
                embedding_start_idx += len(question_candidates)
                
                if best_question:
                    result = {
                        **data,
                        self.output_question_key: best_question["question"]
                    }
                    processed_results.append(result)
                else:
                    self.logger.warning(f"No valid question generated for data: {data[self.input_db_id_key]}")
                    failed_data.append(data)
            else:
                self.logger.warning(f"No question candidates for data: {data[self.input_db_id_key]}")
                failed_data.append(data)
        
        if self.output_question_key in raw_dataframe.columns:
            all_results = existing_data + processed_results
        else:
            all_results = processed_results
        
        final_df = pd.DataFrame(all_results)
        output_file = storage.write(final_df)
        
        self.logger.info(f"Question generation results saved to {output_file}")
        self.logger.info(f"Successfully processed: {len(processed_results)}")
        if failed_data:
            self.logger.warning(f"Failed to generate questions for: {len(failed_data)} entries")
        
        if self.output_question_key in raw_dataframe.columns:
            return []
        else:
            return [self.output_question_key]