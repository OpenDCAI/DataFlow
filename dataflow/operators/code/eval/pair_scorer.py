import pandas as pd
import re
from typing import List, Tuple

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PairScorer(OperatorABC):
    """
    PairScorer is an operator that evaluates the quality of a generated code snippet
    against its source instruction. It uses an LLM to provide both a numerical score
    and textual feedback, acting as an automated code reviewer.
    """

    def __init__(self, llm_serving: LLMServingABC):
        """
        Initializes the operator with a language model serving endpoint.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = (
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
        required_keys = [self.input_instruction_key, self.input_code_key]
        forbidden_keys = [self.output_score_key, self.output_feedback_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for PairScorer: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by PairScorer: {conflict}")

    def _build_prompts(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Builds a list of prompts for the LLM based on the instruction-code pairs.
        """
        prompts = [
            self.prompt_template.format(
                instruction=row[self.input_instruction_key],
                code=row[self.input_code_key]
            )
            for _, row in dataframe.iterrows()
        ]
        return prompts

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

    def run(
        self, 
        storage: DataFlowStorage, 
        input_instruction_key: str = "generated_instruction", 
        input_code_key: str = "generated_code",
        output_score_key: str = "quality_score",
        output_feedback_key: str = "quality_feedback"
    ) -> List[str]:
        """
        Executes the scoring process for instruction-code pairs.
        
        It reads data, prompts an LLM for evaluation, parses the results,
        and writes the data with new score and feedback columns back to storage.
        
        Returns:
            A list containing the names of the newly created output columns.
        """
        self.logger.info("Running PairScorer operator...")
        
        # Store keys for use in helper methods
        self.input_instruction_key = input_instruction_key
        self.input_code_key = input_code_key
        self.output_score_key = output_score_key
        self.output_feedback_key = output_feedback_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Build prompts for the LLM
        self.logger.info(f"Building prompts for scoring from columns '{self.input_instruction_key}' and '{self.input_code_key}'...")
        formatted_prompts = self._build_prompts(dataframe)
        
        # 4. Query the LLM serving endpoint
        self.logger.info(f"Sending {len(formatted_prompts)} requests to LLM for scoring...")
        responses = self.llm_serving.generate_from_input(user_inputs=formatted_prompts, system_prompt="")
        
        # 5. Parse the responses
        parsed_results = [self._parse_score_and_feedback(r) for r in responses]
        scores, feedbacks = zip(*parsed_results) # Unzip list of tuples into two lists
        
        # 6. Add the new data to the DataFrame
        dataframe[self.output_score_key] = scores
        dataframe[self.output_feedback_key] = feedbacks
        
        # 7. Write the results back to storage
        output_file = storage.write(dataframe)
        self.logger.success(f"PairScorer finished. Results with new columns '{self.output_score_key}' and '{self.output_feedback_key}' saved to {output_file}")

        # 8. Return the names of the new columns
        return [self.output_score_key, self.output_feedback_key]