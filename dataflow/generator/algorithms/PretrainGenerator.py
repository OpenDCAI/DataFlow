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
from dataflow.utils.utils import get_logger


@GENERATOR_REGISTRY.register()
class PretrainGenerator:
    def __init__(self, config: Dict):
        self.logger = get_logger()
        self.config = config
        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.key = config['keys']
        self.logger.info(f"Initializing PretrainGenerator with input_file={self.input_file}, output_file={self.output_file}, keys={self.key}...")
        self.model = self.__init_model__()

    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()

        if generator_type == "local":
            self.logger.info("Using LocalModelGenerator...")
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            self.logger.info("Using APIGenerator_aisuite...")
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            self.logger.info("Using APIGenerator_request...")
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")

    def run(self):
        self.logger.info("Running PretrainGenerator...")

        # Load the raw dataframe from the input file
        try:
            raw_dataframe = pd.read_json(self.input_file, lines=True)
            self.logger.info(f"Loaded input file {self.input_file}, number of rows: {len(raw_dataframe)}")
        except Exception as e:
            self.logger.error(f"Error loading input file {self.input_file}: {e}")
            return

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in raw_dataframe.iterrows():
            raw_content = row.get(self.key, '')
            if raw_content:
                llm_input = self._generate_llm_input(raw_content)
                llm_inputs.append(llm_input)

        # Generate the text using the model
        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = self.model.generate_text_from_input(llm_inputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return

        # Add the generated content back to the dataframe
        raw_dataframe['generated_content'] = generated_outputs

        # Save the updated dataframe to the output file
        try:
            raw_dataframe.to_json(self.output_file, orient='records', lines=True)
            self.logger.info(f"Saved the output to {self.output_file}.")
        except Exception as e:
            self.logger.error(f"Error saving the output file {self.output_file}: {e}")

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