from dataflow.prompts.general_text import PretrainGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PretrainGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = PretrainGeneratorPrompt()    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于给定文档内容，生成预训练格式的多轮对话问答数据。" if lang == "zh" else "Generate pre-training format multi-turn dialogue Q&A data based on the given document content."

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "generated_content"):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running PretrainGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in dataframe.iterrows():
            raw_content = row.get(self.input_key, '')
            if raw_content:
                llm_input = self.prompts.pt_generate_prompt(raw_content)
                llm_inputs.append(llm_input)
        
        # Generate the text using the model
        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return

        # Add the generated content back to the dataframe
        dataframe['generated_content'] = generated_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key