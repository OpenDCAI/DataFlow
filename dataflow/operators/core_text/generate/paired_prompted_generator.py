import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PairedPromptedGenerator(OperatorABC):
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
                "PairedPromptedGenerator：基于两列配对输入（input_key_1 与 input_key_2）进行成对提示生成。\n"
                "算子会将 system_prompt 与每行的两列文本按固定模板拼接后，调用 LLM 服务批量生成结果，"
                "并将模型输出写回到 DataFrame 的指定列。\n\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象（实现 LLMServingABC 接口）\n"
                "- system_prompt：系统提示词（默认 'You are a helpful agent.'）。该提示会放在每条样本前缀，"
                "  用于约束模型的角色与输出风格。\n"
                "- input_key_1：第一列输入字段名（默认 'input_key_1'）\n"
                "- input_key_2：第二列输入字段名（默认 'input_key_2'）\n"
                "- output_key：输出字段名（默认 'generated_content'）\n\n"
                "处理逻辑：\n"
                "1) 从 storage 中读取名为 'dataframe' 的 DataFrame；\n"
                "2) 对于每一行，若 input_key_1 与 input_key_2 均非空，则按模板：\n"
                "   system_prompt + input_key_1 + 值 + '\\n' + input_key_2 + 值\n"
                "   构造 LLM 输入；\n"
                "3) 批量调用 llm_serving.generate_from_input 生成文本；\n"
                "4) 将生成结果写入 DataFrame 的 output_key 列并保存。\n\n"
                "输出：\n"
                "- 返回写入了生成结果的新 DataFrame（由 storage 管理保存），\n"
                "- 返回 output_key 以便后续算子引用。"
            )
        elif lang == "en":
            return (
                "PairedPromptedGenerator: generate text from paired inputs (input_key_1 and input_key_2).\n"
                "The operator concatenates the system_prompt with the two input fields per row, calls the LLM "
                "in batch, and writes the outputs to the specified column in the DataFrame.\n\n"
                "Inputs:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- system_prompt: System-level instruction (default 'You are a helpful agent.'). "
                "  It prefixes each sample to guide the model role and style.\n"
                "- input_key_1: Name of the first input field (default 'input_key_1')\n"
                "- input_key_2: Name of the second input field (default 'input_key_2')\n"
                "- output_key: Name of the output field (default 'generated_content')\n\n"
                "Flow:\n"
                "1) Read a DataFrame named 'dataframe' from storage;\n"
                "2) For each row where both input_key_1 and input_key_2 are non-empty, build the LLM prompt as:\n"
                "   system_prompt + input_key_1 + value + '\\n' + input_key_2 + value;\n"
                "3) Call llm_serving.generate_from_input in batch to obtain generations;\n"
                "4) Write generations to the DataFrame under output_key and persist via storage.\n\n"
                "Outputs:\n"
                "- The updated DataFrame (persisted by storage),\n"
                "- The output_key for downstream operator reference."
            )
        else:
            return "Generate text from paired inputs using a system prompt; writes results to the specified DataFrame column."

    def run(self, storage: DataFlowStorage, input_key_1: str = "input_key_1", input_key_2: str = 'input_key_2', output_key: str = "generated_content"):
        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in dataframe.iterrows():
            raw_content_1 = row.get(input_key_1, '')
            raw_content_2 = row.get(input_key_2, '')
            if raw_content_1 and raw_content_2:
                llm_input = self.system_prompt + input_key_1 + str(raw_content_1) + '\n' + input_key_2 + str(raw_content_2)
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
        dataframe[output_key] = generated_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key
