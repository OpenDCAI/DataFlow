import json
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class CodeFilter:
    def __init__(self, config :dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.output_file = config.get("output_file")
        self.filter_key = config.get("filter_key")
        self.pass_value = config.get("pass_value")
        if isinstance(self.pass_value, list):
            self.pass_value = [int(_) for _ in self.pass_value]
        elif isinstance(self.pass_value, str):
            self.pass_value = int(self.pass_value)
        
    def run(self):
        """
        Run the code filter.
        """
        with open(self.input_file, 'r') as f:
            data = [json.loads(_) for _ in f]
        new_data = []
        for item in data:
            if isinstance(self.pass_value, list):
                if item[self.filter_key] in self.pass_value:
                    new_data.append(item)
            else:
                if item[self.filter_key] == int(self.pass_value):
                    new_data.append(item)
        with open(self.output_file, 'w') as f:
            for item in new_data:
                json.dump(item, f)
                f.write('\n')