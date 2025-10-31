import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC 
from dataflow.core import LLMServingABC
from dataflow.prompts.code import CodeInstructionEnhancement, DiyCodePrompt

from typing import Union
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
@prompt_restrict(
    CodeInstructionEnhancement,
    DiyCodePrompt
)
@OPERATOR_REGISTRY.register()
class CodeEnhancementInstructionGenerator(OperatorABC):
    """
    CodeEnhancementInstructionGenerator is an operator that uses an LLM to standardize 
    and normalize instructions into a consistent format for code generation tasks. 
    It rewrites original instructions into standardized English instruction + code block format.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template: Union[CodeInstructionEnhancement, DiyCodePrompt, DIYPromptABC] = None):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        
        # Initialize prompt template
        if prompt_template is None:
            prompt_template = CodeInstructionEnhancement()
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
                "该算子用于增强人类指令，将不同输出格式的任务统一为生成完整函数。\n\n"
                "输入参数：\n"
                "- input_key: 包含原始代码片段的字段名 (默认: 'code')\n"
                "输出参数：\n"
                "- output_key: 用于存储生成指令的字段名 (默认: 'generated_instruction')\n"
            )
        else: 
            return (
                "This operator enhances human instructions by unifying tasks with different "
                "output formats into complete function generation tasks.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the original instruction messages (default: 'messages')\n"
                "Output Parameters:\n"
                "- output_key: Field name to store the enhanced instruction (default: 'generated_instruction')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist and output columns don't conflict.
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
        def get_human_instruction(messages):
            """Extract human instruction from message list."""
            for item in messages:
                if item.get('role') == 'HUMAN':
                    return item.get('content', '')
            return ''
        return [
            self.prompt_template.build_prompt(instruction=get_human_instruction(row[self.input_key]))
            for _, row in dataframe.iterrows()
        ]

    def _parse_instruction(self, response: str) -> str:
        """
        Parse the LLM's raw response to extract the enhanced instruction.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Clean instruction string without extra whitespace
        """
        return response.strip()

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str = "messages", 
        output_key: str = "generated_instruction"
    ) -> List[str]:
        """
        Executes the instruction synthesis process.
        
        Reads data from storage, enhances instructions for each message,
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