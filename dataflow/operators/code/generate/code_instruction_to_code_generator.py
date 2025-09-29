import pandas as pd
import re
from typing import List

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.code import CodeInstructionToCodeGeneratorPrompt, DiyCodePrompt

@OPERATOR_REGISTRY.register()
class CodeInstructionToCodeGenerator(OperatorABC):
    """
    CodeInstructionToCodeGenerator is an operator that takes a natural language instruction and
    uses an LLM to generate a corresponding code snippet. This is the second step
    in a 'self-instruct' style data synthesis pipeline for code.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template=None):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        
        # Initialize prompt template
        if prompt_template is None:
            prompt_template = CodeInstructionToCodeGeneratorPrompt()
        elif isinstance(prompt_template, str):
            prompt_template = DiyCodePrompt(prompt_template)
        self.prompt_template = prompt_template
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子根据给定的人类指令生成相应的代码片段。\n\n"
                "输入参数：\n"
                "- input_key: 包含人类指令的字段名 (默认: 'instruction')\n"
                "输出参数：\n"
                "- output_key: 用于存储生成代码的字段名 (默认: 'generated_code')\n"
            )
        else: # Default to English
            return (
                "This operator generates a code snippet based on a given natural language instruction.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the human instruction (default: 'instruction')\n"
                "Output Parameters:\n"
                "- output_key: Field name to store the generated code (default: 'generated_code')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist and output columns don't.
        """
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _build_prompts(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Builds a list of prompts for the LLM based on the input instructions.
        """
        prompts = [
            self.prompt_template.build_prompt(instruction=row[self.input_key])
            for _, row in dataframe.iterrows()
        ]
        return prompts

    def _parse_code(self, response: str) -> str:
        """
        Parse the LLM's raw response to extract only the code.
        Removes potential markdown code blocks and leading/trailing whitespace.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Clean code string without markdown formatting
        """
        # Use regex to find content within ```python ... ``` or ``` ... ```
        code_block_match = re.search(r"```(?:python\n)?(.*)```", response, re.DOTALL)
        if code_block_match:
            # If a markdown block is found, extract its content
            return code_block_match.group(1).strip()
        else:
            # Otherwise, assume the whole response is code and just strip it
            return response.strip()

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str = "instruction", 
        output_key: str = "generated_code"
    ) -> List[str]:
        """
        Executes the code generation process.
        
        It reads data from storage, generates code for each instruction,
        and writes the updated data back to storage.
        
        Returns:
            A list containing the name of the newly created output column.
        """
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        
        formatted_prompts = self._build_prompts(dataframe)
        responses = self.llm_serving.generate_from_input(formatted_prompts)
        
        codes = [self._parse_code(r) for r in responses]
        dataframe[self.output_key] = codes
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Generated code saved to {output_file}")

        return [self.output_key]