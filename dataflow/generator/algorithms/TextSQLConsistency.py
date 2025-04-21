from dataflow.generator.utils.Prompts import TextSQLConsistencyPrompt
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from tqdm import tqdm
import logging
import os
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class TextSQLConsistency:
    def __init__(self, config: dict):
        '''
        Initialize the TextSQLConsistency with the provided configuration.
        '''
        self.config = config
        self.prompt = TextSQLConsistencyPrompt()
        self.model_generator = self.__init_model__()

        # Input and output file paths and keys
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_sql_key = self.config.get("input_sql_key", "SQL")
        self.input_question_key = self.config.get("input_question_key", "question")
        self.input_evidence_key = self.config.get("input_evidence_key", "")
        self.output_key = self.config.get("output_key", "consistency")
        self.output_reason_key = self.config.get("output_reason_key", "consistency_reason")

        # Ensure required paths and keys are provided
        if not self.input_file or not self.output_file:
            raise ValueError("Both input_file and output_file must be specified in the config.")

    def __init_model__(self):
        '''
        Initialize the model generator based on the configuration.
        '''
        generator_type = self.config.get("generator_type", "local").lower()
        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
        
    def _reformat_prompt(self, dataframe):
        '''
        Reformat the prompts in the dataframe to generate questions.
        '''
        formatted_prompts = []
        if self.input_evidence_key == "":
            for index, row in dataframe.iterrows():
                sql = row[self.input_sql_key]
                question = row[self.input_question_key]
                used_prompt = self.prompt.text_sql_consistency_prompt(question, sql)
                formatted_prompts.append(used_prompt.strip())
        else:
            for index, row in dataframe.iterrows():
                sql = row[self.input_sql_key]
                question = row[self.input_question_key]
                evidence = row[self.input_evidence_key]
                used_prompt = self.prompt.text_sql_consistency_prompt(question, sql, evidence)
                formatted_prompts.append(used_prompt.strip())
        return formatted_prompts
        
    def run(self):
        '''
        Runs the consistency judgement, reading from the input file and saving results to output.
        '''
        # Read input file: only accept jsonl format
        dataframe = pd.read_json(self.input_file, lines=True)
        
        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Reformat the prompts for question generation
        formatted_prompts = self._reformat_prompt(dataframe)

        # Generate responses using the model
        responses = self.model_generator.generate_text_from_input(formatted_prompts)

        for (idx, row), response in zip(dataframe.iterrows(), responses):
            try:
                conclusion = None
                response_lower = response.lower()
                
                if "conclusion:" in response_lower:
                    conclusion_part = response_lower.split("conclusion:")[1].strip()
                    if "analysis:" in response_lower:
                        analysis_part = response_lower.split("conclusion")[0].split("analysis:")[1].strip()
                    else:
                        analysis_part = ""
                else:
                    raise ValueError("Response does not contain 'conclusion:'")
                
                if "no" in conclusion_part:
                    conclusion = False
                elif "yes" in conclusion_part:
                    conclusion = True
                else:
                    raise ValueError("Could not determine conclusion from response")
                
                dataframe.at[idx, self.output_key] = conclusion
                dataframe.at[idx, self.output_reason_key] = analysis_part.strip()
                
            except Exception as e:
                logging.warning(f"Failed to judge the consistency of the SQL: {e}")
                dataframe.at[idx, self.output_key] = "ERROR"
                dataframe.at[idx, self.output_reason_key] = f"Failed to judge: {str(e)}"

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrame to JSON file
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input sql key exists in the dataframe
        if self.input_sql_key not in dataframe.columns:
            raise ValueError(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
        
        # Check if the input question key exists in the dataframe
        if self.input_question_key not in dataframe.columns:
            raise ValueError(f"input_question_key: {self.input_question_key} not found in the dataframe.")
        
        # Check if the input evidence key exists in the dataframe
        if self.input_evidence_key != "" and self.input_evidence_key not in dataframe.columns:
            raise ValueError(f"input_evidence_key: {self.input_evidence_key} not found in the dataframe.")
        
        # Check if the output key already exists in the dataframe
        if self.output_key in dataframe.columns:
            raise ValueError(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output reason key already exists in the dataframe
        if self.output_reason_key in dataframe.columns:
            raise ValueError(f"Found {self.output_reason_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
