import pandas as pd
from typing import List

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC # For type hinting if needed
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class InstructionSynthesizer(OperatorABC):
    """
    InstructionSynthesizer is an operator that uses an LLM to generate a human-readable
    instruction based on a given code snippet. This is the first step in a 
    'self-instruct' style data synthesis pipeline for code.
    """

    def __init__(self, llm_serving: LLMServingABC):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = (
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
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子用于分析代码片段并反向生成可能产生该代码的人类指令。\n\n"
                "输入参数：\n"
                "- input_code_key: 包含原始代码片段的字段名 (默认: 'raw_code')\n"
                "输出参数：\n"
                "- output_instruction_key: 用于存储生成指令的字段名 (默认: 'generated_instruction')\n"
            )
        else: # Default to English
            return (
                "This operator analyzes a code snippet and reverse-engineers a human instruction "
                "that could have produced it.\n\n"
                "Input Parameters:\n"
                "- input_code_key: Field name containing the raw code snippet (default: 'raw_code')\n"
                "Output Parameters:\n"
                "- output_instruction_key: Field name to store the generated instruction (default: 'generated_instruction')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist and output columns don't.
        """
        required_keys = [self.input_code_key]
        forbidden_keys = [self.output_instruction_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for InstructionSynthesizer: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by InstructionSynthesizer: {conflict}")

    def _build_prompts(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Builds a list of prompts for the LLM based on the input code.
        """
        prompts = [
            self.prompt_template.format(code=row[self.input_code_key])
            for _, row in dataframe.iterrows()
        ]
        return prompts

    def _parse_instruction(self, response: str) -> str:
        """
        Parse the LLM's raw response to extract the clean instruction.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Clean instruction string without extra whitespace
        """
        # The prompt is designed to make the LLM output only the instruction.
        # This parsing step is mainly for cleaning up potential whitespace.
        return response.strip()

    def run(
        self, 
        storage: DataFlowStorage, 
        input_code_key: str = "raw_code", 
        output_instruction_key: str = "generated_instruction"
    ) -> List[str]:
        """
        Executes the instruction synthesis process.
        
        It reads data from storage, generates instructions for each code snippet,
        and writes the updated data back to storage.
        
        Returns:
            A list containing the name of the newly created output column.
        """
        self.logger.info("Running InstructionSynthesizer operator...")
        
        # Store keys for use in helper methods
        self.input_code_key = input_code_key
        self.output_instruction_key = output_instruction_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Build prompts for the LLM
        self.logger.info(f"Building prompts from column '{self.input_code_key}'...")
        formatted_prompts = self._build_prompts(dataframe)
        
        # 4. Query the LLM serving endpoint
        self.logger.info(f"Sending {len(formatted_prompts)} requests to LLM...")
        # Assuming the llm_serving object has a method like `generate_from_input` or `query`
        responses = self.llm_serving.generate_from_input(user_inputs=formatted_prompts, system_prompt="")
        
        # 5. Parse the responses
        instructions = [self._parse_instruction(r) for r in responses]
        
        # 6. Add the new data to the DataFrame
        dataframe[self.output_instruction_key] = instructions
        
        # 7. Write the results back to storage for the next operator
        output_file = storage.write(dataframe)
        self.logger.success(f"InstructionSynthesizer finished. Results with new column '{self.output_instruction_key}' saved to {output_file}")

        # 8. Return the name of the new column
        return [self.output_instruction_key]