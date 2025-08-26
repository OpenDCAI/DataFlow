import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "You are a helpful agent."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于用户提供的提示词（prompt）生成数据。结合系统提示词和输入内容生成符合要求的输出文本。"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- system_prompt：系统提示词，定义模型行为，默认为'You are a helpful agent.'\n"
                "- input_key：输入内容字段名，默认为'raw_content'\n"
                "- output_key：输出生成内容字段名，默认为'generated_content'\n"
                "输出参数：\n"
                "- 包含生成内容的DataFrame\n"
                "- 返回输出字段名，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Generate data from user-provided prompts. Combines system prompt and input content to generate desired output text.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- system_prompt: System prompt to define model behavior, default is 'You are a helpful agent.'\n"
                "- input_key: Field name for input content, default is 'raw_content'\n"
                "- output_key: Field name for output generated content, default is 'generated_content'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing generated content\n"
                "- Returns output field name for subsequent operator reference"
            )
        else:
            return (
                "PromptedGenerator generates text based on system prompt and input content."
            )

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "generated_content"):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in dataframe.iterrows():
            raw_content = row.get(self.input_key, '')
            if raw_content:
                llm_input = self.system_prompt + str(raw_content)
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
        dataframe[self.output_key] = generated_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key
