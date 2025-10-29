import pandas as pd
from typing import List

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC # For type hinting if needed
from dataflow.core import LLMServingABC
from dataflow.prompts.code import CodeCodeToInstructionGeneratorPrompt, DiyCodePrompt

from typing import Union
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
@prompt_restrict(
    CodeCodeToInstructionGeneratorPrompt,
    DiyCodePrompt
)
@OPERATOR_REGISTRY.register()
class CodeCodeToInstructionGenerator(OperatorABC):
    """
    CodeCodeToInstructionGenerator is an operator that uses an LLM to generate a human-readable
    instruction based on a given code snippet. This is the first step in a 
    'self-instruct' style data synthesis pipeline for code.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template: Union[CodeCodeToInstructionGeneratorPrompt, DiyCodePrompt, DIYPromptABC] = None):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        
        # Initialize prompt template
        if prompt_template is None:
            prompt_template = CodeCodeToInstructionGeneratorPrompt()
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
                "该算子用于分析代码片段并反向生成可能产生该代码的人类指令。\n\n"
                "输入参数：\n"
                "- input_key: 包含原始代码片段的字段名 (默认: 'code')\n"
                "输出参数：\n"
                "- output_key: 用于存储生成指令的字段名 (默认: 'generated_instruction')\n"
            )
        else: # Default to English
            return (
                "This operator analyzes a code snippet and reverse-engineers a human instruction "
                "that could have produced it.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the raw code snippet (default: 'code')\n"
                "Output Parameters:\n"
                "- output_key: Field name to store the generated instruction (default: 'generated_instruction')\n"
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
        Builds a list of prompts for the LLM based on the input code.
        """
        prompts = [
            self.prompt_template.build_prompt(code=row[self.input_key])
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
        input_key: str = "code", 
        output_key: str = "generated_instruction"
    ) -> List[str]:
        """
        Executes the instruction synthesis process.
        
        It reads data from storage, generates instructions for each code snippet,
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
        
        instructions = [self._parse_instruction(r) for r in responses]
        dataframe[self.output_key] = instructions
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Generated instructions saved to {output_file}")

        return [self.output_key]