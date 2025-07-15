from dataflow.prompts.reasoning import AnswerGeneratorPrompt, GeneralAnswerGeneratorPrompt, DiyAnswerGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from typing import Literal

@OPERATOR_REGISTRY.register()
class AnswerGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self,
                llm_serving: LLMServingABC,
                content_type: Literal["math", "general", "diy"] = "math",
                prompt_template: str = None
                ):
        
        self.logger = get_logger()
        self.prompts = AnswerGeneratorPrompt()    
        self.llm_serving = llm_serving
        self.content_type = content_type
        self.prompt_template = prompt_template
        
        if content_type == "math":
            self.prompts = AnswerGeneratorPrompt()
        elif content_type == "general":
            self.prompts = GeneralAnswerGeneratorPrompt()
        elif content_type == "diy":
            self.prompts = DiyAnswerGeneratorPrompt(self.prompt_template)
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为给定问题生成答案，支持数学、通用和自定义类型的内容，调用大语言模型进行推理。\n\n"
                "输入参数：\n"
                "- llm_serving：LLM服务实例，用于生成答案\n"
                "- content_type：内容类型，可选值为'math'（数学）、'general'（通用）、'diy'（自定义），默认'math'\n"
                "- prompt_template：自定义提示模板字符串，当content_type为'diy'时必填\n\n"
                "输出参数：\n"
                "- output_key：生成的答案字段，默认'generated_cot'"
            )
        elif lang == "en":
            return (
                "This operator generates answers for given questions, supporting math, general, and custom content types using LLMs for reasoning. \n\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving instance for answer generation\n"
                "- content_type: Content type, optional values 'math', 'general', 'diy', default 'math'\n"
                "- prompt_template: Custom prompt template string, required when content_type is 'diy'\n\n"
                "Output Parameters:\n"
                "- output_key: Generated answer field, default 'generated_cot'"
            )
        else:
            return "AnswerGenerator produces answers for questions using large language models."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        questions = dataframe[self.input_key].tolist()
        inputs = [self.prompts.Classic_COT_Prompt(question) for question in questions]

        return inputs

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "instruction", 
        output_key:str = "generated_cot"
        ):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        answers = self.llm_serving.generate_from_input(formatted_prompts)

        dataframe[self.output_key] = answers
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]