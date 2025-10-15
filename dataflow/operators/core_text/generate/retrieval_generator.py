import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.light_rag_serving import LightRAGServing
import asyncio

@OPERATOR_REGISTRY.register()
class RetrievalGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LightRAGServing,
                 system_prompt: str = "You are a helpful agent.",
                 json_schema: dict = None,
                 ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.json_schema = json_schema
        self.system_prompt = system_prompt
    
    def run(self,
            storage: DataFlowStorage,
            input_key: str = "raw_content",
            output_key: str = "generated_content",
            ):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running RetrievalGenerator...")

        # Load the raw dataframe from the input file
        df = storage.read('dataframe')
        self.logger.info(f"Loading, number of tasks: {len(df)}")

        llm_inputs = []
        for index, row in df.iterrows():
            raw_content = row.get(self.input_key, '')
            if raw_content:
                llm_input = str(raw_content)
                llm_inputs.append(llm_input)

        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = asyncio.run(self.llm_serving.generate_from_input(llm_inputs, self.system_prompt))
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return
        
        df[self.output_key] = generated_outputs

        output_file = storage.write(df)
        return output_key