from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.Prompts import CodeRefinerPrompt as CRP
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class CodeRefiner:
    def __init__(self, config :dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.input_key = config.get('input_key', 'content')
        self.output_key = config.get('output_key', 'commented_content')
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
        
    
    def reformat_prompt(self, dataframe : pd.DataFrame):
        """
        Reformat the prompt for the oss inst generator.
        """
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Input key {self.input_key} not found in dataframe columns: {dataframe.columns}")
        
        # get self.input_key from dataframe to list
        input_list = dataframe[self.input_key].to_list()
        # use prompt
        crp = CRP()
        inputs = [crp.code_refiner_prompt(code) for code in input_list]
        return inputs

    def run(self):
        """
        Main method to execute the OssInstGenerator.
        """
        dataframe = pd.read_json(self.input_file,lines=True)
        inputs = self.reformat_prompt(dataframe)
        outputs = self.model.generate_text_from_input(inputs)


        dataframe[self.output_key] = outputs

        # save the output
        dataframe.to_json(self.output_file,orient="records",lines=True)
        