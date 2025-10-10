from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import TreeinstructPrompt  

@OPERATOR_REGISTRY.register()
class TreeinstructSampleEvaluator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'TreeinstructScore'
        self.prompt = TreeinstructPrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通过生成语法树的节点数来衡量指令复杂性，节点越多表示指令越复杂。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_instruction_key：指令字段名\n"
                "- output_key：输出得分字段名，默认'TreeinstructScore'\n"
                "输出参数：\n"
                "- 包含指令复杂性得分的DataFrame"
            )
        elif lang == "en":
            return (
                "Measure instruction complexity by syntax tree size; more nodes mean more complexity.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_instruction_key: Field name for instruction\n"
                "- output_key: Field name for output score, default 'TreeinstructScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing instruction complexity scores"
            )
        else:
            return "Measure instruction complexity by syntax tree size; more nodes mean more complexity."
    
    def get_score(self, samples, input_instruction_key):
        system_prompts = []
        user_prompts = []
        for sample in samples:
            instruction = sample.get(input_instruction_key, [''])
            system_prompts.append(self.prompt.build_system_prompt(instruction))
            user_prompts.append(self.prompt.build_user_prompt())

        inputs = [system + "\n" + user for system, user in zip(system_prompts, user_prompts)]
        responses = self.llm_serving.generate_from_input(user_inputs=inputs)
        
        scores = []
        for response in responses:
            response_lines = response.strip().split("\n")
            score_line = response_lines[-1]
            score = float(score_line.split()[0])
            scores.append(score)
            
        return scores

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str):
        self.logger.info(f"Evaluating {self.score_name}...")
        samples = dataframe.to_dict(orient='records')
        scores = self.get_score(samples, input_instruction_key)
        self.logger.info("Evaluation complete!")
        return scores


    def run(self, storage: DataFlowStorage, input_instruction_key: str, output_key: str='TreeinstructScore'):
        self.input_instruction_key = input_instruction_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_instruction_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
