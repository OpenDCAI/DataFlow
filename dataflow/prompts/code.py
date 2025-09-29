'''
A collection of prompts for the code operators.
'''

class CodeQualityEvaluatorPrompt:
    '''
    The prompt for the code quality evaluator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, instruction: str, code: str) -> str:
        """
        Generate system prompt for code quality evaluation.
        """
        prompt = (
            "You are a meticulous and critical code reviewer. Your task is to evaluate the quality of the "
            "provided 'Generated Code' based on the given 'Instruction'.\n\n"
            "Provide a single integer score from 1 (poor) to 10 (excellent) and brief, constructive feedback. "
            "Your entire response MUST strictly follow the format below.\n\n"
            "Instruction: {instruction}\n\n"
            "Generated Code:\n"
            "```python\n"
            "{code}\n"
            "```\n\n"
            "Evaluation Criteria:\n"
            "1. **Correctness & Completeness**: Does the code accurately and fully implement the instruction? Does it handle obvious edge cases?\n"
            "2. **Clarity & Best Practices**: Is the code clean, readable, and does it follow standard conventions (e.g., PEP 8 for Python)?\n"
            "3. **Efficiency**: Is the implementation reasonably efficient for the given task?\n\n"
            "Format your response EXACTLY as follows:\n"
            "Score: [integer score from 1 to 10]\n"
            "Feedback: [your feedback here]"
        )
        return prompt.format(instruction=instruction, code=code)


class CodeCodeToInstructionGeneratorPrompt:
    '''
    The prompt for the code to instruction generator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, code: str) -> str:
        """
        Generate system prompt for code to instruction generation.
        """
        prompt = (
            "You are an expert programmer and a clear communicator. Your task is to analyze the "
            "provided code snippet and generate a single, concise, and natural human instruction "
            "that could have produced this code.\n\n"
            "The instruction should be a directive, like 'Write a function that...' or 'Create a class to...'. "
            "Do NOT add any explanations, comments, or markdown formatting. Output only the instruction text.\n\n"
            "Code Snippet:\n"
            "```\n"
            "{code}\n"
            "```\n\n"
            "Generated Instruction:"
        )
        return prompt.format(code=code)


class CodeInstructionToCodeGeneratorPrompt:
    '''
    The prompt for the instruction to code generator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, instruction: str) -> str:
        """
        Generate system prompt for instruction to code generation.
        """
        prompt = (
            "You are a world-class coding assistant. Your task is to fulfill the following request precisely. "
            "Your response must contain ONLY the code that satisfies the instruction. "
            "Do not add any explanations, introductory sentences, or markdown formatting like ```python ... ```.\n\n"
            "Request: {instruction}\n\n"
            "Generated Code:"
        )
        return prompt.format(instruction=instruction)


class DiyCodePrompt:
    '''
    The prompt for custom code operations.
    '''
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def build_prompt(self, **kwargs) -> str:
        """
        Generate prompt using custom template.
        """
        try:
            return self.prompt_template.format(**kwargs)
        except Exception as e:
            # If formatting fails, return the original template
            return self.prompt_template
