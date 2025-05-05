import json
import os
import pandas as pd
from dataflow.generator.utils import LocalModelGenerator, APIGenerator_aisuite, APIGenerator_request
from dataflow.generator.utils.Prompts import QuestionDifficultyPrompt
import re
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class QuestionDifficultyClassifier():
    def __init__(self, args):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.config = args
        self.prompts = QuestionDifficultyPrompt()
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_key = self.config.get("input_key", "question")  # default key for question input
        self.output_key = self.config.get("output_key", "classification_result")  # default output key
        
        # Ensure input_file and output_file are provided
        if not self.input_file or not self.output_file:
            raise ValueError("Both input_file and output_file must be specified in the config.")

        # Initialize the model
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

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        # Check if input_key is in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {key_list}")

        formatted_prompts = []
        for text in dataframe[self.input_key]:
            used_prompt = self.prompts.question_synthesis_prompt(text)
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self):
        # read input file : accept jsonl file only
        dataframe = pd.read_json(self.input_file,lines=True)
        # model = self.__init_model__()
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.model.generate_text_from_input(formatted_prompts)

        rating_scores = []
        for response in responses:
            match = re.search(r'Rating:\s*([\d.]+)', response)
            score = float(match.group(1)) if match else None
            rating_scores.append(score)

        #if self.output_key in dataframe.columns:
        #    key_list = dataframe.columns.tolist()
        #    raise ValueError(f"Found {self.output_text_key} in the dataframe, which leads to overwriting the existing column, please check the output_text_key: {key_list}")
        
        dataframe[self.output_key] = rating_scores
        
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

            # Save DataFrame to the output file
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

