from dataflow import get_logger
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
'''
A collection of prompts for the diy reasoning operator.
'''
@PROMPT_REGISTRY.register()
class DiyAnswerGeneratorPrompt(PromptABC):
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.logger = get_logger()
        
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template + question + r'''Your response must start directly with "Solution:" without any preamble. Finish your response immediately after the solution.'''
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")

@PROMPT_REGISTRY.register()
class DiyQuestionFilterPrompt(PromptABC):
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.logger = get_logger()
        
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(question=question)
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")
            
@PROMPT_REGISTRY.register()
class DiyQuestionSynthesisPrompt(PromptABC):
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.logger = get_logger()
        
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(question=question)
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")