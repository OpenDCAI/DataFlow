import pandas as pd
import re
from typing import List

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class CodeGenerator(OperatorABC):
    """
    CodeGenerator is an operator that takes a natural language instruction and
    uses an LLM to generate a corresponding code snippet. This is the second step
    in a 'self-instruct' style data synthesis pipeline for code.
    """

    def __init__(self, llm_serving: LLMServingABC):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = (
            "You are a world-class coding assistant. Your task is to fulfill the following request precisely. "
            "Your response must contain ONLY the code that satisfies the instruction. "
            "Do not add any explanations, introductory sentences, or markdown formatting like ```python ... ```.\n\n"
            "Request: {instruction}\n\n"
            "Generated Code:"
        )
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子根据给定的人类指令生成相应的代码片段。\n\n"
                "输入参数：\n"
                "- input_instruction_key: 包含人类指令的字段名 (默认: 'generated_instruction')\n"
                "输出参数：\n"
                "- output_code_key: 用于存储生成代码的字段名 (默认: 'generated_code')\n"
            )
        else: # Default to English
            return (
                "This operator generates a code snippet based on a given natural language instruction.\n\n"
                "Input Parameters:\n"
                "- input_instruction_key: Field name containing the human instruction (default: 'generated_instruction')\n"
                "Output Parameters:\n"
                "- output_code_key: Field name to store the generated code (default: 'generated_code')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist and output columns don't.
        """
        required_keys = [self.input_instruction_key]
        forbidden_keys = [self.output_code_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for CodeGenerator: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by CodeGenerator: {conflict}")

    def _build_prompts(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Builds a list of prompts for the LLM based on the input instructions.
        """
        prompts = [
            self.prompt_template.format(instruction=row[self.input_instruction_key])
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
        input_instruction_key: str = "generated_instruction", 
        output_code_key: str = "generated_code"
    ) -> List[str]:
        """
        Executes the code generation process.
        
        It reads data from storage, generates code for each instruction,
        and writes the updated data back to storage.
        
        Returns:
            A list containing the name of the newly created output column.
        """
        self.logger.info("Running CodeGenerator operator...")
        
        # Store keys for use in helper methods
        self.input_instruction_key = input_instruction_key
        self.output_code_key = output_code_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Build prompts for the LLM
        self.logger.info(f"Building prompts from column '{self.input_instruction_key}'...")
        formatted_prompts = self._build_prompts(dataframe)
        
        # 4. Query the LLM serving endpoint using generate_from_input
        self.logger.info(f"Sending {len(formatted_prompts)} requests to LLM for code generation...")
        responses = self.llm_serving.generate_from_input(user_inputs=formatted_prompts, system_prompt="")
        
        # 5. Parse the responses to extract clean code
        codes = [self._parse_code(r) for r in responses]
        
        # 6. Add the new data to the DataFrame
        dataframe[self.output_code_key] = codes
        
        # 7. Write the results back to storage for the next operator
        output_file = storage.write(dataframe)
        self.logger.success(f"CodeGenerator finished. Results with new column '{self.output_code_key}' saved to {output_file}")

        # 8. Return the name of the new column
        return [self.output_code_key]