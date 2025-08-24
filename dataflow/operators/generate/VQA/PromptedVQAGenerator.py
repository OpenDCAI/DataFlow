from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedVQAGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "You are a helpful assistant."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于视觉问答生成，接收包含图像和问题的输入内容，使用大语言模型生成回答，"
                "并将生成的回答保存到数据框中。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- system_prompt：系统提示词，用于定义模型行为，默认为'You are a helpful assistant.'\n"
                "- input_key：输入内容的字段名，默认为'raw_content'\n"
                "- output_key：输出生成内容的字段名，默认为'generated_content'\n"
                "输出参数：\n"
                "- 返回输出字段名，用于后续算子引用\n"
                "- 在数据框中添加包含生成回答的新列"
            )
        elif lang == "en":
            return (
                "This operator generates visual question answering responses. It receives input content containing images and questions, "
                "uses a large language model to generate answers, and saves the generated answers to the dataframe.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- system_prompt: System prompt to define model behavior, default is 'You are a helpful assistant.'\n"
                "- input_key: Field name for input content, default is 'raw_content'\n"
                "- output_key: Field name for output generated content, default is 'generated_content'\n\n"
                "Output Parameters:\n"
                "- Returns output field name for subsequent operator reference\n"
                "- Adds a new column containing generated answers to the dataframe"
            )
        else:
            return (
                "PromptedVQAGenerator processes visual questions and generates answers using a large language model."
            )
    
    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "generated_content"):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running Prompted VQA Generator...")

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
        self.logger.info("Prompted VQA Generator done")
        return output_key