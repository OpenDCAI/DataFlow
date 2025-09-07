from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import AlpagasusPrompt  

@OPERATOR_REGISTRY.register()
class AlpagasusSampleEvaluator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, dimension: str = 'quality'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'AlpagasusScore'
        self.dimension = dimension
        self.prompt = AlpagasusPrompt(dimension=self.dimension)
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通过调用GPT评估指令的质量，返回一个质量得分，得分越高表明指令的质量越高。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- dimension：评估维度，默认为'quality'\n"
                "- input_instruction_key：指令字段名\n"
                "- input_input_key：输入文本字段名\n"
                "- input_output_key：输出文本字段名\n"
                "- output_key：输出得分字段名，默认'AlpagasusScore'\n"
                "输出参数：\n"
                "- 包含评估得分的DataFrame"
            )
        elif lang == "en":
            return (
                "Evaluate instruction quality using GPT; higher scores indicate better quality.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- dimension: Evaluation dimension, default 'quality'\n"
                "- input_instruction_key: Field name for instruction\n"
                "- input_input_key: Field name for input text\n"
                "- input_output_key: Field name for output text\n"
                "- output_key: Field name for output score, default 'AlpagasusScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing evaluation scores"
            )
        else:
            return "Evaluate instruction quality using GPT; higher scores indicate better quality."

    def get_score(self, samples, input_instruction_key, input_input_key, input_output_key):
        system_prompts = []
        user_prompts = []
        for sample in samples:
            instruction = sample.get(input_instruction_key, [''])
            response = sample.get(input_output_key, [''])
            input_text = sample.get(input_input_key, [''])
            system_prompts.append(self.prompt.build_system_prompt(instruction, input_text, response))
            user_prompts.append(self.prompt.build_user_prompt())
        inputs = [system + "\n" + user for system, user in zip(system_prompts, user_prompts)]
        responses = self.llm_serving.generate_from_input(user_inputs=inputs)
        scores = []
        for response in responses:
            score_line = response.strip().split("\n")[0]
            score = float(score_line.split()[0])
            scores.append(score)
            
        return scores

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str, input_input_key: str, input_output_key: str):
        samples = dataframe.to_dict(orient='records')
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = self.get_score(samples, input_instruction_key, input_input_key, input_output_key)
        self.logger.info("Evaluation complete!")
        return scores


    def run(self, storage: DataFlowStorage, input_instruction_key: str, input_input_key: str, input_output_key: str, output_key: str='AlpagasusScore'):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_instruction_key, self.input_input_key, self.input_output_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
