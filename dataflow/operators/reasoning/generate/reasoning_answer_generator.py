from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt

@OPERATOR_REGISTRY.register()
class ReasoningAnswerGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self,
                llm_serving: LLMServingABC,
                prompt_template = None,
                ):
        
        self.logger = get_logger()
        
        if prompt_template is None:
            prompt_template = MathAnswerGeneratorPrompt()
        self.prompts = prompt_template
        self.llm_serving = llm_serving
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为给定问题生成答案，调用大语言模型进行推理。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务实例，用于生成答案\n"
                "- prompt_template：提示模板对象，用于构建生成提示词\n"
                "输出参数：\n"
                "- output_key：生成的答案字段，默认'generated_cot'"
            )
        elif lang == "en":
            return (
                "This operator generates answers for given questions using LLMs for reasoning. \n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving instance for answer generation\n"
                "- prompt_template: Prompt template object for constructing generation prompts\n"
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
        inputs = [self.prompts.build_prompt(question) for question in questions]

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