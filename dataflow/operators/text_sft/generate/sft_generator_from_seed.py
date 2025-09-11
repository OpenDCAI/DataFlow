from dataflow.prompts.general_text import SFTGeneratorSeedPrompt
import re
import json
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving import LocalModelLLMServing_vllm

def extract_json_object(model_output):
    """提取第一个包含 instruction 和 output 字段的 JSON 对象"""
    json_pattern = r'\{[^}]*\}'
    matches = re.findall(json_pattern, model_output)
    for match in matches:
        try:
            obj = json.loads(match)
            if 'instruction' in obj and 'output' in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None

@OPERATOR_REGISTRY.register()
class SFTGeneratorSeed(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, custom_prompt: str):
        self.logger = get_logger()
        self.prompts = SFTGeneratorSeedPrompt(custom_prompt=custom_prompt)    
        self.llm_serving = llm_serving
        self.max_tokens = 4096  
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于给定文档内容，生成监督微调格式的问答数据。并支持用户自定义生成内容要求。从原始文档中提取信息，生成符合SFT格式的指令-响应对。"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- custom_prompt：用户自定义提示词\n"
                "- input_key：输入文档内容字段名，默认为'raw_content'\n"
                "- max_tokens：生成文本的最大token数，默认为4096\n"
                "输出参数：\n"
                "- 包含'instruction'、'output'和'raw_content'字段的DataFrame\n"
                "- 返回包含'instruction'和'output'字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Generate supervised fine-tuning format Q&A data based on the given document content and support user-defined content generation requirements. \n"
                "Extracts information from raw documents to generate instruction-response pairs in SFT format.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- custom_prompt: User-defined custom prompt\n"
                "- input_key: Field name for input document content, default is 'raw_content'\n"
                "- max_tokens: Maximum number of tokens for generated text, default is 4096\n\n"
                "Output Parameters:\n"
                "- DataFrame containing 'instruction', 'output', and 'raw_content' fields\n"
                "- List containing 'instruction' and 'output' field names for subsequent operator reference"
            )
        else:
            return (
                "SFTGeneratorSeed generates SFT format Q&A data from document content with custom prompt support."
            )

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content"):
        self.input_key = input_key
        self.logger.info("Running SFTGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in dataframe.iterrows():
            raw_content = row.get(self.input_key, '')
            llm_input = self.prompts.sft_generate_prompt(content=raw_content)
            llm_inputs.append(llm_input)
        
        # Generate the text using the model
        try:
            self.logger.info("Generating text using the model...")
            outputs = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return

        valid_records = []
        for idx, output in enumerate(outputs):
            result = extract_json_object(output)
            if result:
                result["raw_content"] = dataframe[self.input_key].iloc[idx]  # 添加原文内容
                valid_records.append(result)

        # Add the generated content back to the dataframe
        output_df = pd.DataFrame(valid_records)

        # Save the updated dataframe to the output file
        output_file = storage.write(output_df)
        return ['instruction', 'output']
