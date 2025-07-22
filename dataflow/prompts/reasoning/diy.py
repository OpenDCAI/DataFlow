'''
A collection of prompts for the diy reasoning operator.
'''

class DiyAnswerGeneratorPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
    
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template + question + r'''Your response must start directly with "Solution:" without any preamble. Finish your response immediately after the solution.'''
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")

class DiyQuestionFilterPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
    
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(question=question)
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")
            
class DiyQuestionSynthesisPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
    
    def build_prompt(self, question: str) -> str:
        try:
            return self.prompt_template.format(question=question)
        except:
            self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")