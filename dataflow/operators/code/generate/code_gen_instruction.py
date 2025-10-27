import pandas as pd
from typing import List
import random

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC 
from dataflow.core import LLMServingABC
from dataflow.prompts.code import CodeInstructionGenerate, DiyCodePrompt

@OPERATOR_REGISTRY.register()
class CodeInstructionGenerator(OperatorABC):
    """
    CodeInstructionGenerator is an operator that leverages a Large Language Model to generate 
    human-readable instructions based on few-shot examples sampled from a data pool. The operator 
    creates new instructions that are similar in difficulty and style to the provided examples. 
    This is a critical step in a 'self-instruct' style data synthesis pipeline, designed to expand 
    and enhance instruction datasets for programming tasks.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template=None, num_few_shot: int = 3, num_generate: int = 10):
        """
        Initializes the operator with a language model serving endpoint.
        
        Args:
            llm_serving: LLM serving instance
            prompt_template: Custom prompt template (optional)
            num_few_shot: Number of few-shot examples to use (default: 3)
        """
        self.logger = get_logger()
        self.num_generate = num_generate
        self.llm_serving = llm_serving
        self.num_few_shot = num_few_shot
        self.prompt_template = CodeInstructionGenerate()
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子用于生成新的指令,从数据池中随机抽取few-shot样本,生成类似难度的指令。\n\n"
                "输入参数:\n"
                "- input_key: 包含原始指令的字段名 (默认: 'prompt')\n"
                "输出参数:\n"
                "- output_key: 用于存储生成指令的字段名 (默认: 'generated_instruction')\n"
            )
        else:
            return (
                "This operator generates new instructions by sampling few-shot examples from the data pool "
                "to create instructions of similar difficulty.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the original instructions (default: 'prompt')\n"
                "Output Parameters:\n"
                "- output_key: Field name to store the generated instruction (default: 'generated_instruction')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist.
        """
        required_keys = [self.input_key]

        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        
        if len(dataframe) < self.num_few_shot:
            raise ValueError(f"数据池样本数量({len(dataframe)})少于few-shot数量({self.num_few_shot})")

    def _sample_few_shot_examples(self, dataframe: pd.DataFrame) -> List[dict]:
        """
        andomly sample few-shot examples from the data pool.
        
        Args:
            dataframe: data pool
            
        Returns:
            List of few-shot examples with instruction
        """
        num_samples = min(self.num_few_shot, len(dataframe))
        sampled_indices = random.sample(range(len(dataframe)), num_samples)
        
        few_shot_examples = []
        for idx in sampled_indices:
            row = dataframe.iloc[idx]
            instruction = row[self.input_key]
            few_shot_examples.append({
                'instruction': instruction,
            })
        
        return few_shot_examples

    def _build_prompts(self, dataframe: pd.DataFrame, num_generate: int) -> List[str]:
        """
        构建指定数量的prompt,每个prompt包含随机抽取的few-shot样本
        
        Args:
            dataframe: Data pool
            num_generate: The number of prompts to be generated.
            
        Returns:
            List of prompts
        """
        prompts = []
        for i in range(num_generate):
            few_shot_examples = self._sample_few_shot_examples(dataframe)
            prompt = self.prompt_template.build_prompt(
                few_shot_examples=few_shot_examples
            )
            prompts.append(prompt)
        
        return prompts

    def _parse_instruction(self, response: str) -> str:
        """
        Parse the LLM's raw response to extract the clean instruction.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Clean instruction string without extra whitespace
        """
        return response.strip()

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str = "prompt", 
        output_key: str = "generated_instruction",
    ) -> List[str]:
        """
        Executes the instruction synthesis process.
        Reads data from the data pool, generates the specified number of new instructions, and saves them to a new DataFrame.
        
        Args:
            storage: DataFlow storage instance
            input_key: Field name containing the original instructions
            output_key: Field name to store generated instructions
        
        Returns:
            A list containing the name of the output column.
        """
        self.input_key = input_key
        self.output_key = output_key
        
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        random.seed(42)

        formatted_prompts = self._build_prompts(dataframe, self.num_generate)

        responses = self.llm_serving.generate_from_input(formatted_prompts)

        instructions = [self._parse_instruction(r) for r in responses]

        new_dataframe = pd.DataFrame({
            self.output_key: instructions
        })

        output_file = storage.write(new_dataframe)
        self.logger.info(f"Generated {len(instructions)} new instructions with {self.num_few_shot} few-shot examples each")
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]