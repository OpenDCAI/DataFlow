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
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "两阶段优化指令回复质量：第一阶段调用API生成对回复的评论，第二阶段利用评论调用API改写回复，提升指令对质量。通过迭代优化提高问答对的整体质量。"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_instruction_key：输入指令字段名，默认为'instruction'\n"
                "- input_output_key：输入回复字段名，默认为'output'\n"
                "输出参数：\n"
                "- 包含优化后回复的DataFrame\n"
                "- 返回包含优化后回复字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Two-stage optimization of instruction-response quality: First stage calls API to generate critique on responses, \n"
                "second stage uses critique to call API to refine responses, improving the quality of QA pairs through iterative optimization.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_instruction_key: Field name for input instructions, default is 'instruction'\n"
                "- input_output_key: Field name for input responses, default is 'output'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing refined responses\n"
                "- List containing refined response field name for subsequent operator reference"
            )
        else:
            return (
                "CondorRefiner improves QA pair quality through two-stage critique and refinement process."
            )

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
