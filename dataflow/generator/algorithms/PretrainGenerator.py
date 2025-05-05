import json
import logging
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import torch, os, itertools, string
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.Prompts import PretrainPrompt


@GENERATOR_REGISTRY.register()
class PretrainGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.key = config['keys']
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

    def run(self):
        # Load the raw dataframe from the input file
        raw_dataframe = pd.read_json(self.input_file, lines=True)
        
        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in raw_dataframe.iterrows():
            raw_content = row.get(self.key, '')
            if raw_content:
                llm_input = self._generate_llm_input(raw_content)
                llm_inputs.append(llm_input)
        # Generate the text using the model
        generated_outputs = self.model.generate_text_from_input(llm_inputs)

        # Add the generated content back to the dataframe
        raw_dataframe['generated_content'] = generated_outputs
        
        # Save the updated dataframe to the output file
        raw_dataframe.to_json(self.output_file, orient='records', lines=True)

    def _generate_llm_input(self, raw_content: str) -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template.
        """
        prompt = """
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. 
        Convert the following paragraph into a conversational format with multiple tags of "Question:" followed by "Answer:":
        You can only output as the given format:
        Question: xxx Answer: xxx
        Question: xxx Answer: xxx
        Now please covert the content below.
        {content}
        """
        return prompt.format(content=raw_content)