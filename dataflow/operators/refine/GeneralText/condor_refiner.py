import json
import random
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import CondorPrompt

@OPERATOR_REGISTRY.register()
class CondorRefiner(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.prompt = CondorPrompt()  # 创建 CondorPrompt 类的实例
        self.logger.info(f'{self.__class__.__name__} initialized.')

    def generate_critique(self, question, answer):
        # 批量生成 Critique
        critique_prompts = [self.prompt.create_critique_prompt(q, a) for q, a in zip(question, answer)]
        critique_responses = self.llm_serving.generate_from_input(critique_prompts)
        return critique_responses

    def generate_refined_answer(self, question, answer, critique):
        # 批量生成修改后的答案
        refine_prompts = [self.prompt.create_refine_prompt(q, a, c) for q, a, c in zip(question, answer, critique)]
        refined_answers = self.llm_serving.generate_from_input(refine_prompts)
        refined_answers = [answer.replace('[Improved Answer Start]', '').replace('[Improved Answer End]', '').strip() for answer in refined_answers]
        return refined_answers

    def run(self, storage: DataFlowStorage, input_instruction_key: str='instruction', input_output_key: str='output'):
        df = storage.read('dataframe')
        # 从 storage 获取批量问题和答案
        questions = df.get(input_instruction_key).to_list()
        answers = df.get(input_output_key).to_list()
        # 生成 Critique
        critique_responses = self.generate_critique(questions, answers)
        self.logger.info(f'Generated Critiques for the answers.')

        # 生成修改后的答案
        refined_answers = self.generate_refined_answer(questions, answers, critique_responses)
        self.logger.info(f'Refined Answers generated.')
        df[input_output_key] = refined_answers
        output_file = storage.write(df)
        self.logger.info(f'Refined answers updated in storage.')

        return [input_output_key]
