import random
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class ExtractLines:
    def __init__(self, config: dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.input_key = config.get("input_key")
        self.output_key = config.get("output_key")

    def _load_inputs(self):
        """Load code strings from input JSONL file"""
        dataframe = pd.read_json(self.input_file,lines=True)
        return dataframe, dataframe[self.input_key].tolist()
    
    def _extract_continuous_lines(self, code_string):
        """Extract random continuous lines from a code string"""
        lines = code_string.splitlines()
        if not lines:
            return ""
            
        num_lines_to_extract = random.randint(4, 15)
        start_line = random.randint(0, max(0, len(lines) - num_lines_to_extract))
        return "\n".join(lines[start_line:start_line + num_lines_to_extract])

    def run(self):
        """
        Extract random continuous lines from code strings
        """
        dataframe, code_strings = self._load_inputs()
        extracted_lines = [self._extract_continuous_lines(code) for code in code_strings]
        dataframe[self.output_key] = extracted_lines
        dataframe.to_json(self.output_file,orient="records",lines=True)
        return

