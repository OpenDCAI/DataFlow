from dataflow.prompts.reasoning.math import MathQuestionDifficultyPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core.prompt import prompt_restrict
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import re

@prompt_restrict(
    MathQuestionDifficultyPrompt
)

@OPERATOR_REGISTRY.register()
class ReasoningQuestionDifficultySampleEvaluator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        """
        Initialize the ReasoningCategoryDatasetEvaluator with the provided configuration.
        """
        self.logger = get_logger()
        self.prompts = MathQuestionDifficultyPrompt()
        self.llm_serving = llm_serving

    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于评估问题的难度等级。"
                "通过大语言模型分析问题复杂度，输出1-10级的难度评分。\n\n"
                "输入参数：\n"
                "- input_key：输入问题字段名，必填，无默认值\n"
                "- output_key：写入难度评分的输出字段名，默认为'difficulty_score'\n\n"
                "输出参数：\n"
                "- output_key：从LLM回复中“Rating: X”解析出的浮点难度评分（1-10），无法解析时为-1；若该字段已存在于输入中则报错\n"
                "- 不过滤任何行，所有输入字段保留\n"
                "- 返回包含output_key的列表"
            )
        elif lang == "en":
            return (
                "Evaluates question difficulty level using LLM analysis. "
                "Outputs numerical difficulty score from 1 to 10.\n\n"
                "Input Parameters:\n"
                "- input_key: Input question field name, required, no default\n"
                "- output_key: Name of the output field that receives the difficulty score, default is 'difficulty_score'\n\n"
                "Output Parameters:\n"
                "- output_key: float difficulty score (1-10) parsed from 'Rating: X' in the LLM reply, -1 if it cannot be parsed; raises an error if this field already exists in the input\n"
                "- No rows are filtered; all input fields are kept\n"
                "- Returns a list containing output_key"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str = "instruction", output_key: str = "difficulty_score"):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

        
    def _reformat_prompt(self, dataframe, input_key: str = "instruction") -> list:
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        formatted_prompts = []
        for i, text in enumerate(dataframe[input_key]):
            if text is not None:
                used_prompt = self.prompts.build_prompt(text)
            else:
                used_prompt = None
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self, storage:DataFlowStorage, input_key: str, output_key:str="difficulty_score") -> None:
        """
        Run the question difficulty classification process.
        """
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(
            dataframe,
            input_key=self.input_key,
            output_key=self.output_key
        )
        formatted_prompts = self._reformat_prompt(dataframe, input_key=self.input_key)
        responses = self.llm_serving.generate_from_input(user_inputs=formatted_prompts)

        rating_scores = []
        for response in responses:
            match = re.search(r'Rating:\s*((\d+\.\d+)|\d+)', response)
            if match:
                score_str = match.group(1).rstrip('.')
                try:
                    score = float(score_str)
                except ValueError:
                    score = -1
            else:
                score = -1
            rating_scores.append(score)
        dataframe[output_key] = rating_scores

        output_file = storage.write(dataframe)
        self.logger.info(f"Classification results saved to {output_file}")
        
        return [output_key]