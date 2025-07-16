from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import AlpagasusPrompt  

@OPERATOR_REGISTRY.register()
class AlpagasusScorer(OperatorABC):
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
        return "通过调用 GPT 评估指令的质量，返回一个质量得分，得分越高表明指令的质量越高。" if lang == "zh" else "Evaluate instruction quality using GPT; higher scores indicate better quality."

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
