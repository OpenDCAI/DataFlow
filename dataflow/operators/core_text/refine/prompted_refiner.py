import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedRefiner(OperatorABC):
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
                "PromptedRefiner 根据给定的 system_prompt 对指定列的文本进行改写/润色/规范化，"
                "并将结果**就地写回**同一列（覆盖原内容）。其做法是对每一行拼接 "
                "`system_prompt + raw_content` 作为模型输入，批量生成改写结果。\n"
                "\n输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- system_prompt：系统提示词，用于描述改写目标与风格（默认 'You are a helpful agent.'）\n"
                "- input_key：要改写的文本列名（默认 'raw_content'），改写后会覆盖该列\n"
                "\n输出参数：\n"
                "- 覆盖后的 DataFrame（同名列被改写后的文本）\n"
                "- 无返回值（结果已通过 DataFlowStorage 写出）\n"
                "\n备注：\n"
                "- 该算子**覆盖** input_key 列；若需保留原文，建议先拷贝到新列。\n"
                "- 期望每行在 input_key 列提供可用文本；空值将不会生成对应输入，如与行数不匹配可能导致赋值报错。"
            )
        elif lang == "en":
            return (
                "PromptedRefiner rewrites/refines/normalizes text in a specified column **in place**, "
                "using a provided system_prompt. For each row it concatenates "
                "`system_prompt + raw_content` as the model input and generates the refined text.\n"
                "\nInput Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- system_prompt: Instruction describing the rewrite goal/style (default 'You are a helpful agent.')\n"
                "- input_key: Column to refine (default 'raw_content'); the refined output **overwrites** this column\n"
                "\nOutput:\n"
                "- DataFrame with the same column overwritten by refined text\n"
                "- No return value (the result is written via DataFlowStorage)\n"
                "\nNotes:\n"
                "- This operator **overwrites** the input_key column; copy it first if you need to keep originals.\n"
                "- Each row is expected to provide text in input_key; missing/empty rows won’t form inputs, which may cause "
                "length-mismatch errors on assignment."
            )
        else:
            return (
                "PromptedRefiner rewrites a chosen column in place using `system_prompt + raw_content` as input."
            )


    def run(self, storage: DataFlowStorage, input_key: str = "raw_content"):
        self.input_key = input_key
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
        dataframe[self.input_key] = generated_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return 
