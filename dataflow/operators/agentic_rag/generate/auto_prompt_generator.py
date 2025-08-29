from dataflow.prompts.agenticrag import AutoPromptGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage  
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class AutoPromptGenerator(OperatorABC):
    '''
    AutoPromptGenerator is a class that generates prompts for given document fragments to generate seed QA pairs.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = AutoPromptGeneratorPrompt()    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为给的的文档片段生成提示词，用于生成种子QA对\n\n"
                "输入参数：\n"
                "- input_key: 包含文档片段的字段名\n"
                "- output_key: 包含提示词的字段名\n"
            )
        elif lang == "en":
            return (
                "This operator generates prompts for given document fragments to generate seed QA pairs.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the content\n"
                "- output_key: Field name containing the generated prompt\n"
            )
        else:
            return "AutoPromptGenerator generates prompts for given document fragments to generate seed QA pairs."

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
        inputs = [self.prompts.auto_prompt_generator_prompt(question) for question in questions]

        return inputs

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "text", 
        output_key:str = "generated_prompt"
        ):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        answers = self.llm_serving.generate_from_input(user_inputs=formatted_prompts, system_prompt="")

        dataframe[self.output_key] = answers
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]