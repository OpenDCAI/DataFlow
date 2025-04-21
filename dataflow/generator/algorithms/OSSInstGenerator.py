import os
from ..utils.LocalModelGenerator import LocalModelGenerator
from ..utils.APIGenerator_aisuite import APIGenerator_aisuite
from ..utils.APIGenerator_request import APIGenerator_request
from ..utils.Prompts import OssInstGeneratorPrompt as OIP
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class OSSInstGenerator:
    def __init__(self, config :dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.input_key = config.get("input_key")
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
        
    def parse_llm_output(self, llm_output, key_list = ["Problem", "Analysis", "Solution"]):
        """
        Parses an LLM output string by identifying lines containing 'Problem', 'Analysis', and 'Solution'
        as pivots, then extracts and formats the content into a dictionary.
        """
        lines = llm_output.splitlines()
        
        # Initialize an empty dictionary to hold the results
        parsed_data = {}
        
        # Create a list to store the start index for each key section
        section_start = {key: None for key in key_list}
        
        # Iterate over lines to find the start indices of each section
        for i, line in enumerate(lines):
            for key in key_list:
                if key in line and section_start[key] is None:
                    section_start[key] = i
        
        # If any key has not been found, return None
        if any(start is None for start in section_start.values()):
            return None

        # Now, extract the content for each key from the lines
        for i, key in enumerate(key_list):
            start_index = section_start[key] + 1
            if i + 1 < len(key_list):
                end_index = section_start[key_list[i + 1]]
            else:
                end_index = len(lines)
            
            # Join the lines for the section and strip unnecessary whitespace
            parsed_data[key] = "\n".join(lines[start_index:end_index]).strip()

        return parsed_data
    
    def reformat_prompt(self, dataframe : pd.DataFrame):
        """
        Reformat the prompt for the oss inst generator.
        """
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Input key {self.input_key} not found in dataframe columns: {dataframe.columns}")
        
        # get self.input_key from dataframe to list
        input_list = dataframe[self.input_key].to_list()
        # use prompt
        oip = OIP()
        inputs = [oip.oss_inst_generator_prompt(code) for code in input_list]
        return inputs

    def run(self):
        """
        Main method to execute the OssInstGenerator.
        """
        dataframe = pd.read_json(self.input_file,lines=True)
        inputs = self.reformat_prompt(dataframe)
        outputs = self.model.generate_text_from_input(inputs)


        dataframe[self.output_key] = outputs
        # parse the output
        dataframe[self.output_key] = dataframe[self.output_key].apply(self.parse_llm_output)
        # save the output
        dataframe.to_json(self.output_file,orient="records",lines=True)
        