import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import string

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict, PromptABC, DIYPromptABC
from typing import Union, Any, Set

from dataflow.prompts.core_text import FormatStrPrompt
@prompt_restrict(
    FormatStrPrompt,
)
@OPERATOR_REGISTRY.register()
class FormatStrPromptedGenerator(OperatorABC):
    def __init__(
            self,
            llm_serving: LLMServingABC, 
            system_prompt: str =  "You are a helpful agent.",
            prompt_template: Union[FormatStrPrompt, DIYPromptABC] = FormatStrPrompt,
            json_schema: dict = None,
        ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.json_schema = json_schema
        if prompt_template is None:
            raise ValueError("prompt_template cannot be None")

    def run(
            self, 
            storage: DataFlowStorage,
            output_key: str = "generated_content",
            **input_keys: Any
        ):
        self.storage: DataFlowStorage = storage
        self.output_key = output_key
        self.logger.info("Running PromptTemplatedGenerator...")
        self.input_keys = input_keys
        need_fields = set(input_keys.keys())
    # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        llm_inputs = []

        for idx, row in dataframe.iterrows():
            key_dict = {}
            for key in need_fields:
                key_dict[key] = row[input_keys[key]]
            prompt_text = self.prompt_template.build_prompt(need_fields, **key_dict)
            llm_inputs.append(prompt_text)
        self.logger.info(f"Prepared {len(llm_inputs)} prompts for LLM generation.")
        # Create a list to hold all generated contents
        # Generate content using the LLM serving
        generated_outputs = self.llm_serving.generate_from_input(user_inputs = llm_inputs, system_prompt = self.system_prompt, json_schema = self.json_schema)

        dataframe[self.output_key] = generated_outputs

        output_file = self.storage.write(dataframe)
        return output_key
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于模板化提示词（Prompt Template）生成内容的算子。"
                "该算子使用用户定义的提示模板（StrFormatPrompt 或 DIYPrompt），"
                "结合输入数据中的字段自动构造完整提示词并调用大语言模型生成结果。\n\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口，用于执行文本生成任务\n"
                "- prompt_template：提示词模板对象（StrFormatPrompt 或 DIYPromptABC），用于定义提示结构\n"
                "- input_keys：输入字段映射字典，用于将DataFrame中的列名映射到模板字段\n"
                "- output_key：输出生成内容字段名，默认为'generated_content'\n\n"
                "输出参数：\n"
                "- 包含生成结果的新DataFrame\n"
                "- 返回输出字段名，以便后续算子引用\n\n"
                "使用场景：\n"
                "适用于需要通过模板化提示构建多样输入、批量生成文本内容的场景，例如标题生成、摘要生成、问答模板填充等。"
            )
        elif lang == "en":
            return (
                "An operator for content generation based on templated prompts. "
                "This operator uses a user-defined prompt template (StrFormatPrompt or DIYPromptABC) "
                "to automatically construct full prompts from input data fields and generate outputs via an LLM.\n\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface, responsible for text generation\n"
                "- prompt_template: Prompt template object (StrFormatPrompt or DIYPromptABC) defining the prompt structure\n"
                "- input_keys: Dictionary mapping DataFrame column names to template fields\n"
                "- output_key: Field name for generated content, default is 'generated_content'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing generated outputs\n"
                "- Returns the output field name for downstream operator reference\n\n"
                "Use Case:\n"
                "Ideal for tasks requiring templated prompt-driven generation, such as title generation, text summarization, or Q&A filling."
            )
        else:
            return (
                "PromptTemplatedGenerator generates text based on a user-defined prompt template."
            )