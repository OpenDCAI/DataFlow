import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.general_text import SFTFromScratchGeneratorPrompt

@prompt_restrict(
    SFTFromScratchGeneratorPrompt
)


@OPERATOR_REGISTRY.register()
class RandomDomainKnowledgeRowGenerator(OperatorABC):
    def __init__(
    self, 
    llm_serving: LLMServingABC,  
    prompt_template : SFTFromScratchGeneratorPrompt,
    generation_num : int,
    domain_keys : str
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.generation_num = generation_num
        self.domain_keys = domain_keys
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        def get_desc(lang: str = "zh"):
            if lang == "zh":
                return (
                    "RandomDomainKnowledgeRowGenerator算子用于结合提示模板(prompt_template)与LLM服务对象(llm_serving)，"
                    "批量生成与指定领域相关的文本内容。\n\n"
                    "功能说明：\n"
                    "- 结合SFTFromScratchGeneratorPrompt模板，根据domain_keys随机选择领域并生成内容；\n"
                    "- 当输入DataFrame为空时，可通过generation_num参数控制生成样本数量；\n"
                    "- 生成的文本结果将写入指定字段(output_key)，并返回该字段名供后续算子使用。\n\n"
                    "参数说明：\n"
                    "- llm_serving：LLM服务对象，需实现LLMServingABC接口；\n"
                    "- prompt_template：提示模板实例，需为SFTFromScratchGeneratorPrompt类型；\n"
                    "- storage：DataFlowStorage对象，用于读取与写入数据；\n"
                    "- output_key：生成结果写入的字段名，默认为'generated_content'；\n"
                    "- generation_num：生成内容数量，默认为1；\n"
                    "- domain_keys：指定或限制生成内容所属领域。\n\n"
                    "输出说明：\n"
                    "- 返回值：输出字段名(output_key)，供后续算子引用；\n"
                    "- 同时将包含生成内容的新DataFrame写回至存储。"
                )

            elif lang == "en":
                return (
                    "The RandomDomainKnowledgeRowGenerator operator generates domain-related text content "
                    "by combining a prompt template (prompt_template) with an LLM serving instance (llm_serving).\n\n"
                    "Function Description:\n"
                    "- Utilizes the SFTFromScratchGeneratorPrompt template to randomly select domains via domain_keys;\n"
                    "- Supports content generation when no input DataFrame is available, controlled by generation_num;\n"
                    "- Generated text is written to the specified output field (output_key), and the field name is returned.\n\n"
                    "Parameter Description:\n"
                    "- llm_serving: LLM serving object implementing the LLMServingABC interface;\n"
                    "- prompt_template: Prompt template instance of type SFTFromScratchGeneratorPrompt;\n"
                    "- storage: DataFlowStorage object used for reading and writing data;\n"
                    "- output_key: Name of the field to write generated results (default: 'generated_content');\n"
                    "- generation_num: Number of contents to generate when there is no input data (default: 1);\n"
                    "- domain_keys: Domain key(s) specifying or constraining the generation domain; empty string for random.\n\n"
                    "Output Description:\n"
                    "- Returns the output field name (output_key) for downstream reference;\n"
                    "- Writes the DataFrame containing generated content back to storage."
                )
            else:
                return (
                    "RandomDomainKnowledgeRowGenerator算子用于结合提示模板(prompt_template)与LLM服务对象(llm_serving)，批量生成领域文本内容。"
                )


    def run(self, storage: DataFlowStorage, output_key: str = "generated_content"):
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
        for i in range(self.generation_num):
            llm_inputs.append(self.prompt_template.build_prompt(self.domain_keys))
            
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
