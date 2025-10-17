
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
'''
A collection of prompts for model evaluation.
'''

@PROMPT_REGISTRY.register()
class AnswerJudgePrompt(PromptABC):
    """
    用于构建答案评判的提示词模板
    """
    def __init__(self):
        pass
    
    def build_prompt(self, answer, reference_answer):
        prompt = f"""
        As an answer evaluation expert, please assess whether the following answer is correct.
        
        Reference Answer: {reference_answer}
        
        Current Answer: {answer}
        
        Please carefully analyze whether the current answer is semantically consistent with the reference answer. 
        Focus only on comparing the answers themselves, not on how the problem is solved.
        Don't just look at the surface text, understand the essential content of the answers.
        If the current answer is semantically consistent with the reference answer, even if expressed differently, it should be judged as correct.
        
        Please return your judgment result in JSON format:
        {{"judgement_result": true}} indicates the answer is correct
        {{"judgement_result": false}} indicates the answer is incorrect
        
        Your judgment:
        """
        return prompt

@PROMPT_REGISTRY.register()
class AnswerJudgePromptQuestion(PromptABC):
    """
    用于构建答案评判的提示词模板
    """
    def __init__(self):
        pass
    
    def build_prompt(self, question, answer, reference_answer):
        prompt = f"""
        As an answer evaluation expert, please assess whether the following answer is correct.
        
        Question: {question}
        
        Reference Answer: {reference_answer}
        
        Current Answer: {answer}
        
        Please carefully analyze whether the current answer is semantically consistent with the reference answer. 
        Focus only on comparing the answers themselves, not on how the problem is solved.
        Don't just look at the surface text, understand the essential content of the answers.
        If the current answer is semantically consistent with the reference answer, even if expressed differently, it should be judged as correct.
        
        Please return your judgment result in JSON format:
        {{"judgement_result": true}} indicates the answer is correct
        {{"judgement_result": false}} indicates the answer is incorrect
        
        Your judgment:
        """
        return prompt