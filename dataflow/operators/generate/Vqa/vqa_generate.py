from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class VqaGenerate(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "generated_content"):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running VqaGenerate...")

        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        llm_inputs = []
        for index, row in dataframe.iterrows():
            raw_content = row.get(self.input_key, '')
            if raw_content:
                llm_inputs.append(raw_content)
        
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs, self.system_prompt)
        
        dataframe[self.output_key] = llm_outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Saving to {output_file}")
        self.logger.info("VqaGenerate done")
        return output_key