from dataflow.prompts.general_text import SupervisedFinetuneGeneratorPrompt
import re
import json
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

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

from transformers import AutoTokenizer  # 引入 tokenizer 库

@OPERATOR_REGISTRY.register()
class SupervisedFinetuneGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = SupervisedFinetuneGeneratorPrompt()    
        self.llm_serving = llm_serving
        
        self.tokenizer = llm_serving.tokenizer
        self.max_tokens = 4096  
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于给定文档内容，生成监督微调格式的问答数据。" if lang == "zh" else "Generate supervised fine-tuning format Q&A data based on the given document content."

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content"):
        self.input_key = input_key
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
                # 对 raw_content 进行 tokenization
                tokens = self.tokenizer.encode(raw_content, truncation=True, max_length=self.max_tokens)
                truncated_content = self.tokenizer.decode(tokens, skip_special_tokens=True)
                # 使用截断后的内容生成 prompt
                llm_input = self.prompts.sft_generate_prompt(content=truncated_content)
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
