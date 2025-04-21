import os
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.Prompts import CodeScorerPrompt as CSP
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class CodeScorer:
    def __init__(self, config :dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.input_key = config.get("input_key")
        self.input_key_for_problem_description = config.get("input_key_for_problem_description")
        self.input_key_for_analysis = config.get("input_key_for_analysis")
        self.input_key_for_solution = config.get("input_key_for_solution")
        self.output_key = config.get("output_key")
        self.model = self.__init_model__()

    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()

        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
        
    def process_prompt(self, dataframe :pd.DataFrame):
        """
        Process the prompt for the code scorer.
        """
        inputs = []
        sys_prompt = CSP().code_scorer_prompt()
        for index, row in dataframe.iterrows():
            oss_inst = row[self.input_key]
            content = sys_prompt + "[Problem Description]\n" + oss_inst[self.input_key_for_problem_description] + "\n" + "[Analysis]\n" + oss_inst[self.input_key_for_analysis] + "\n" + "[Solution]\n" + oss_inst[self.input_key_for_solution]
            inputs.append(content)
        return inputs
    
    def run(self):
        """
        Run the code scorer.
        """
        dataframe = pd.read_json(self.input_file,lines=True)
        inputs = self.process_prompt(dataframe)
        scores = self.model.generate_text_from_input(inputs)
        dataframe[self.output_key] = scores
        dataframe.to_json(self.output_file,orient="records",lines=True)