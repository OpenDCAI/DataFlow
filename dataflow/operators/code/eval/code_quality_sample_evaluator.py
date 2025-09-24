import pandas as pd
import re
from typing import List, Tuple

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.code import CodeQualityEvaluatorPrompt, DiyCodePrompt

@OPERATOR_REGISTRY.register()
class CodeQualitySampleEvaluator(OperatorABC):
    """
    CodeQualitySampleEvaluator is an operator that evaluates the quality of a generated code snippet
    against its source instruction. It uses an LLM to provide both a numerical score
    and textual feedback, acting as an automated code reviewer.
    """

    def __init__(self, llm_serving: LLMServingABC, prompt_template=None):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'CodeQualityScore'
        
        # Initialize prompt template
        if prompt_template is None:
            prompt_template = CodeQualityEvaluatorPrompt()
        elif isinstance(prompt_template, str):
            prompt_template = DiyCodePrompt(prompt_template)
        self.prompt_template = prompt_template
        
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子用于评估生成的代码片段与其源指令的匹配质量，并输出分数和反馈。\n\n"
                "输入参数：\n"
                "- input_instruction_key: 包含人类指令的字段名 (默认: 'generated_instruction')\n"
                "- input_code_key: 包含生成代码的字段名 (默认: 'generated_code')\n"
                "输出参数：\n"
                "- output_score_key: 用于存储质量分数的字段名 (默认: 'quality_score')\n"
                "- output_feedback_key: 用于存储质量反馈的字段名 (默认: 'quality_feedback')\n"
            )
        else: # Default to English
            return (
                "This operator evaluates the quality of a generated code snippet against its source instruction, providing a score and feedback.\n\n"
                "Input Parameters:\n"
                "- input_instruction_key: Field name containing the human instruction (default: 'generated_instruction')\n"
                "- input_code_key: Field name containing the generated code (default: 'generated_code')\n"
                "Output Parameters:\n"
                "- output_score_key: Field name to store the quality score (default: 'quality_score')\n"
                "- output_feedback_key: Field name to store the quality feedback (default: 'quality_feedback')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure required columns exist and output columns don't.
        """
        required_keys = [self.input_key]
        forbidden_keys = [self.output_score_key, self.output_feedback_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for CodeQualitySampleEvaluator: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by CodeQualitySampleEvaluator: {conflict}")

    def _build_prompts(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Builds a list of prompts for the LLM based on the instruction-code pairs.
        """
        prompts = []
        for _, row in dataframe.iterrows():
            pair = row[self.input_key]
            if isinstance(pair, dict):
                instruction = pair.get('instruction', '')
                code = pair.get('code', '')
            else:
                instruction = str(pair)
                code = str(pair)
            
            prompt = self.prompt_template.build_prompt(instruction=instruction, code=code)
            prompts.append(prompt)
        return prompts

    def _score_func(self, instruction: str, code: str) -> Tuple[int, str]:
        """
        Evaluate a single instruction-code pair and return score and feedback.
        
        Args:
            instruction: The instruction text
            code: The generated code text
            
        Returns:
            Tuple of (score, feedback) where score is an integer and feedback is a string
        """
        prompt = self.prompt_template.build_prompt(instruction=instruction, code=code)
        response = self.llm_serving.generate_from_input(user_inputs=[prompt], system_prompt="")
        
        if not response or len(response) == 0:
            self.logger.warning("Empty response from LLM")
            return 0, "No response from LLM"
            
        return self._parse_score_and_feedback(response[0])
    
    def _parse_score_and_feedback(self, response: str) -> Tuple[int, str]:
        """
        Parse the LLM's raw response to extract the score and feedback.
        Handles potential formatting errors gracefully.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Tuple of (score, feedback) where score is an integer and feedback is a string
        """
        try:
            score_match = re.search(r"Score:\s*(\d+)", response)
            feedback_match = re.search(r"Feedback:\s*(.*)", response, re.DOTALL)
            
            score = int(score_match.group(1)) if score_match else 0
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."
            
            return score, feedback
        except (AttributeError, ValueError, IndexError):
            # If parsing fails for any reason, return default error values
            self.logger.warning(f"Failed to parse LLM evaluation output: '{response}'")
            return 0, "Failed to parse LLM evaluation output."

    def eval(self, dataframe: pd.DataFrame, input_key: str) -> Tuple[List[int], List[str]]:
        """
        Evaluate instruction-code pairs and return scores and feedbacks.
        
        Args:
            dataframe: Input DataFrame
            input_key: Field name containing instruction-code pairs (as dict with 'instruction' and 'code' keys)
            
        Returns:
            Tuple of (scores, feedbacks) lists
        """
        self.logger.info(f"Evaluating {self.score_name}...")
        
        scores = []
        feedbacks = []
        
        for _, row in dataframe.iterrows():
            pair = row[input_key]
            if isinstance(pair, dict):
                instruction = pair.get('instruction', '')
                code = pair.get('code', '')
            else:
                # Try to get instruction and code from separate columns
                instruction = row.get('generated_instruction', str(pair))
                code = row.get('generated_code', str(pair))
            
            score, feedback = self._score_func(instruction, code)
            scores.append(score)
            feedbacks.append(feedback)
        
        self.logger.info("Evaluation complete!")
        return scores, feedbacks
    
    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str,
        output_score_key: str = "quality_score",
        output_feedback_key: str = "quality_feedback"
    ):
        """
        Executes the scoring process for instruction-code pairs.
        
        Args:
            storage: Data storage object
            input_key: Field name containing instruction-code pairs
            output_score_key: Field name for quality scores
            output_feedback_key: Field name for quality feedback
        """
        self.input_key = input_key
        self.output_score_key = output_score_key
        self.output_feedback_key = output_feedback_key
        
        dataframe = storage.read("dataframe")
        scores, feedbacks = self.eval(dataframe, input_key)
        
        dataframe[self.output_score_key] = scores
        dataframe[self.output_feedback_key] = feedbacks
        storage.write(dataframe)