from dataflow.core.prompt import PromptABC
from dataflow.utils.registry import PROMPT_REGISTRY
'''
A collection of prompts for the code operators.
'''
@PROMPT_REGISTRY.register()
class CodeQualityEvaluatorPrompt(PromptABC):
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
            "1. **Correctness & Completeness**: Does the code accurately and fully implement the instruction? Does it handle obvious edge cases? Are all necessary imports included (e.g., List, Dict, Optional from typing, other required modules)?\n"
            "2. **Clarity & Best Practices**: Is the code clean, readable, and does it follow standard conventions (e.g., PEP 8 for Python)?\n"
            "3. **Efficiency**: Is the implementation reasonably efficient for the given task?\n\n"
            "Format your response EXACTLY as follows:\n"
            "Score: [integer score from 1 to 10]\n"
            "Feedback: [your feedback here]"
        )
        return prompt.format(instruction=instruction, code=code)

@PROMPT_REGISTRY.register()
class CodeCodeToInstructionGeneratorPrompt(PromptABC):
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
@PROMPT_REGISTRY.register()
class CodeInstructionGeneratePrompt(PromptABC):
    '''
    The prompt for generating new instructions based on few-shot examples.
    '''
    def __init__(self):
        pass

    def build_prompt(self, few_shot_examples) -> str:
        """
        Generate prompt for creating new instructions similar to the few-shot examples.
        """
        examples_text = ""
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"Example {i}:\n{example['instruction']}\n\n"
        
        prompt = (
            "You are tasked with generating a NEW programming instruction similar in difficulty and style to the provided examples.\n\n"
            "Output MUST follow EXACTLY this format (no extra text before/after):\n"
            "Please provide a self-contained Python script that solves the following problem in a markdown code block\n"
            "```\\n"
            "[optional imports if needed]\\n"
            "\\n"
            "\\n"
            "def function_name(...)-> ReturnType:\\n"
            "    \"\"\" Problem description derived from the original instruction.\\n"
            "    Include input/output description and constraints if any.\\n"
            "    Provide at least one doctest example:\\n"
            "    >>> function_name(example_input)\\n"
            "    expected_output\\n"
            "    \"\"\"\\n"
            "```\\n"
            "GIVEN EXAMPLES:\n"
            f"{examples_text}"
            "REQUIREMENTS:\n"
            "1. Generate ONE new instruction that is similar in difficulty and complexity to the examples above\n"
            "2. Make it diverse - do not simply copy or slightly modify the examples\n"
            "3. The instruction should be clear, specific, and solvable\n"
            "4. Maintain similar level of detail and specificity as the examples\n"
            "NEW INSTRUCTION:"
        )
        return prompt
@PROMPT_REGISTRY.register()
class CodeInstructionEnhancement(PromptABC):
    '''
    The prompt for instruction standardization and enhancement.
    Converts original instructions into a standardized format with proper Python function templates.
    '''
    def __init__(self):
        pass

    def build_prompt(self, instruction: str) -> str:
        """
        Generate system prompt for instruction normalization.
        Only require the output instruction to be about a Python function.
        """
        prompt = (
            "Rewrite the ORIGINAL INSTRUCTION into a standardized English instruction + code block.\n"
            "Output MUST follow EXACTLY this format (no extra text before/after):\n"
            "Please provide a self-contained Python script that solves the following problem in a markdown code block\n"
            "```\\n"
            "[optional imports if needed]\\n"
            "\\n"
            "\\n"
            "def function_name(...)-> ReturnType:\\n"
            "    \"\"\" Problem description derived from the original instruction.\\n"
            "    Include input/output description and constraints if any.\\n"
            "    Provide at least one doctest example:\\n"
            "    >>> function_name(example_input)\\n"
            "    expected_output\\n"
            "    \"\"\"\\n"
            "```\\n"
            "REQUIREMENTS:\n"
            "1. The first line (sentence) must be exactly: Please provide a self-contained Python script that solves the following problem in a markdown code block\n"
            "2. The code fence uses raw ``` (no language tag). Nothing outside the fence except the first sentence.\n"
            "3. Inside the fence: optionally add needed imports (omit if unnecessary), then TWO blank lines, then ONE function.\n"
            "4. Infer a concise snake_case function name from the original instruction.\n"
            "5. Provide full type annotations for parameters and return value (use reasonable types; if uncertain use str / int / List[str] etc.).\n"
            "6. The function body MUST contain ONLY the docstring (no pass, no implementation, no other statements).\n"
            "7. Docstring must be English, multi-line, and include: problem description, input/output description, constraints (if any), and at least one doctest derived or plausibly inferred.\n"
            "8. Do NOT add additional functions, classes, comments, blank sections, placeholders (no TODO, no ...).\n"
            "9. Do NOT echo the original instruction verbatim if it contains formatting artifacts—clean it while preserving meaning.\n"
            "10. Absolutely no extra explanatory text outside the specified output format.\n"
            "ORIGINAL INSTRUCTION:\n{instruction}\n"
            "Produce ONLY the final standardized instruction + code block per the rules."
        )
        return prompt.format(instruction=instruction)
@PROMPT_REGISTRY.register()
class CodeInstructionToCodeGeneratorPrompt(PromptABC):
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

@PROMPT_REGISTRY.register()
class DiyCodePrompt(PromptABC):
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
