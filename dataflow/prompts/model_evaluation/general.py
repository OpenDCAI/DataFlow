
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
    
    def build_prompt(self, answer, reference_answer, question=None):
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
    
@PROMPT_REGISTRY.register()
class AnswerJudgeMultipleQuestionsPrompt(PromptABC):
    """
    用于构建答案评判的提示词模板，支持多个子问题的判断。
    """
    def __init__(self):
        pass
    
    def build_prompt(self, answer, reference_answer, question=None):
        prompt = f"""
        As an answer evaluation expert, please assess whether the following answer is correct.
        
        Question: {question}
        
        Reference Answer: {reference_answer}
        
        Current Answer: {answer}
        
        Please carefully analyze whether the current answer is semantically consistent with the reference answer. 
        Focus only on comparing the answers themselves, not on how the problem is solved.
        Don't just look at the surface text, understand the essential content of the answers.
        If the current answer is semantically consistent with the reference answer, even if expressed differently, it should be judged as correct.
        
        The question may contain multiple sub-questions (e.g., ①②③ or (a)(b), etc.).
        You should first identify the sub-questions in the question, then evaluate the correctness of each corresponding part in the current answer.
        Your output should be a JSON array, where each element is "true" or "false" (use string instead of boolean), indicating whether the answer to each sub-question is correct.
        If there is only one question, also return a single-element array.
        
        If the reference answer is incomplete so that you are not able to judge some subquestions, mark the corresponding sub-questions as "empty".
        
        Example:
        Question: ① 1+2=? ② What is 2+2? ③ What is 3+3?
        Reference Answer: ① 3 ③ 6
        Current Answer: ① Three ② Four ③ Seven
        Output: ["true", "empty", "false"]
        
        
        Your judgment:
        """
        return prompt