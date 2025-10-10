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
    
class CodeInstructionEnhancement:
    '''
    The prompt for the code to instruction generator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, instruction: str) -> str:
        """
        Generate system prompt for instruction normalization.
        Only require the output instruction to be about a Python function.
        """
        prompt = (
            "You are an expert programmer and a clear communicator. "
            "Your task is to read the following instruction and rewrite ONLY the directive part, "
            "so that it becomes a directive for implementing a Python function. "
            "The rewritten directive must be a single sentence, starting with 'Write a complete Python function that ...', "
            "and must directly describe the main problem to be solved in Python function terms.\n"
            "Do NOT add any explanations, comments, or markdown formatting. Output only the instruction text.\n"
            "If the original instruction contains problem descriptions, constraints, input/output formats, examples, or explanations, DO NOT change or rewrite those parts—preserve all such content as-is after your rewritten directive.\n"
            "If the original instruction specifies an output format, IGNORE the output format requirement in your rewrite (do not include it in the directive sentence).\n\n"
            "Original Instruction:\n"
            "'''\n"
            "{instruction}\n"
            "'''\n\n"
            "Normalized Instruction:\n"
            "Write a Python function that ...\n"
            "[All original problem description, constraints, input/output formats, examples, and explanations follow here, unchanged]"
        )
        return prompt.format(instruction=instruction)


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

class GenHumanevalDatasetInstructionPrompt:
    '''
    The prompt for generating synthetic benchmark samples similar to Humaneval.
    '''
    def __init__(self):
        pass

    def build_prompt(self, reference_sample: str) -> str:
        """
        Generate prompt for creating function samples with similar documentation style.
        
        Args:
            reference_sample: Function definition with docstring (including type hints and examples)
        
        Returns:
            Formatted generation prompt with flexible content constraints
        """
        prompt = (
            "You are a code instruction generation expert. Create new instructions that "
            "strictly follow the formatting style of the reference sample but with "
            "different functionality. Follow these rules:\n\n"
            "1. Formatting Requirements:\n"
            "   - Preserve all type hints (change types if needed for new functionality)\n"
            "   - Maintain identical docstring structure:\n"
            "     * First line: Clear one-line description ending with period\n"
            "     >>> doctests: Include at least 2 test cases showing usage\n"
            "   - Include ALL necessary imports (e.g., `from typing import List`)\n"
            "2. Content Variation Rules:\n"
            "   - Change the function's purpose completely\n"
            "   - Parameters and return type can vary as needed\n"
            "   - Maintain similar complexity level\n"
            "   - Ensure doctests demonstrate edge cases\n\n"
            "3. Output Instructions:\n"
            "   - ONLY output the new function definition\n"
            "   - No additional explanations or comments\n\n"
            "Reference Sample:\n"
            "------\n"
            "{reference_sample}\n"
            "------\n\n"
            "Generated Function:"
        )
        return prompt.format(reference_sample=reference_sample)
    
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
