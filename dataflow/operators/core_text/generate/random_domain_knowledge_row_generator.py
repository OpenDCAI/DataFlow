import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()

class RandomDomainKnowledgeRowGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "You are a helpful agent.", user_prompt: str = "{}"):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RandomDomainKnowledgeRowGenerator算子用于结合系统提示词(system_prompt)和用户自定义提示模板(user_prompt)，批量生成领域知识相关文本内容。\n"
                "注意：本算子随机生成的SFT数据所参考的领域是人为预先设定的，具体领域列表可参考prompt中的domain_keys。\n"
                "核心功能：\n"
                "- 支持无输入数据时，按generation_num参数生成指定数量的内容；\n"
                "- 支持自定义system_prompt和user_prompt，user_prompt可通过'{}'占位符灵活插入输入内容或整行字典；\n"
                "- 支持指定输出字段(output_key)\n"
                "参数说明：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口；\n"
                "- system_prompt：系统提示词，定义模型行为，默认为'You are a helpful agent.'；\n"
                "- user_prompt：用户提示词模板，默认为'{}'，可通过'{}'占位符插入输入内容或整行字典；\n"
                "- output_key：输出生成内容字段名，默认为'generated_content'；\n"
                "- generation_num：无输入数据时生成内容的数量，默认为1；\n"
                "输出：\n"
                "- 包含生成内容的DataFrame；\n"
                "- 返回输出字段名(output_key)，供后续算子引用。"
            )
        elif lang == "en":
            return (
                "RandomDomainKnowledgeRowGenerator operator is used to batch generate domain knowledge related text by combining a system prompt (system_prompt) and a user-defined prompt template (user_prompt).\n"
                "Note: The domains referenced for randomly generated SFT data are manually predefined, see the domain_keys in the prompt for the specific list.\n"
                "Main features:\n"
                "- Supports generating a specified number of outputs according to the generation_num parameter when there is no input data;\n"
                "- Supports custom system_prompt and user_prompt, user_prompt can flexibly insert input content or the entire row dict via '{}' placeholder;\n"
                "- Supports specifying the output field (output_key)\n"
                "Parameter description:\n"
                "- llm_serving: LLM serving object, must implement LLMServingABC interface;\n"
                "- system_prompt: System prompt, defines model behavior, default is 'You are a helpful agent.';\n"
                "- user_prompt: User prompt template, default is '{}', can insert input content or row dict via '{}' placeholder;\n"
                "- output_key: Output field name for generated content, default is 'generated_content';\n"
                "- generation_num: Number of outputs to generate when there is no input data, default is 1;\n"
                "Output:\n"
                "- DataFrame containing generated content;\n"
                "- Returns output field name (output_key) for subsequent operator reference."
            )
        else:
            return (
                "RandomDomainKnowledgeRowGenerator算子用于结合系统提示词(system_prompt)和用户自定义提示模板(user_prompt)，批量生成领域知识相关文本内容。"
            )

    def run(self, storage: DataFlowStorage, output_key: str = "generated_content", generation_num: int = 1):
        """
        主流程：基于输入数据和提示词生成文本内容。

        参数说明：
        - storage: DataFlowStorage对象，用于读写数据；
        - output_key: 输出字段名，默认为'generated_content'；
        - generation_num: 生成内容的数量，默认为1；

        返回：
        - 输出字段名（output_key），供后续算子引用。
        """
        self.output_key = output_key
        self.logger.info("Running RandomDomainKnowledgeRowGenerator...")

        # 从存储中读取DataFrame
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded data, number of rows: {len(dataframe)}")

        llm_inputs = []
        
        # 按generation_num生成指定数量的输入
        for i in range(generation_num):
            llm_input = self.system_prompt + self.user_prompt
            llm_inputs.append(llm_input)
            
        try:
            self.logger.info("Generating text using the model...")
            # 调用LLM服务生成文本
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return

        # 将生成的内容写入DataFrame新列
        dataframe[self.output_key] = generated_outputs

        # 将结果写回存储
        output_file = storage.write(dataframe)
        return output_key
