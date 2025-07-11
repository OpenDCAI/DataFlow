from dataflow.prompts.reasoning import AnswerGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class AnswerGenerator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = AnswerGeneratorPrompt()    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于生成数学问题的标准答案，调用大语言模型进行分步推理和计算。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_file：输出文件路径\n"
                "- generator_type：生成器类型（aisuite/request）\n"
                "- model_name：使用的大模型名称\n"
                "- max_worker：并发线程数\n\n"
                "输出参数：\n"
                "- output_key：生成的答案字段"
            )
        elif lang == "en":
            return (
                "This operator generates standard answers for math problems using LLMs "
                "for step-by-step reasoning and calculation.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_file: Output file path\n"
                "- generator_type: Generator type (aisuite/request)\n"
                "- model_name: Name of the model used\n"
                "- max_worker: Number of threads\n\n"
                "Output Parameters:\n"
                "- output_key: Generated answer field"
            )
        else:
            return "AnswerGenerator produces standardized answers for mathematical questions."

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