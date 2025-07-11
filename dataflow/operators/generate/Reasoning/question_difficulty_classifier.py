from dataflow.prompts.reasoning import QuestionDifficultyPrompt
import pandas as pd
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class QuestionDifficultyClassifier(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.logger = get_logger()
        self.prompts = QuestionDifficultyPrompt()
        self.llm_serving = llm_serving

    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于评估问题的难度等级。"
                "通过大语言模型分析问题复杂度，输出1-10级的难度评分。\n\n"
                "输入参数：\n"
                "- eval_stage：评估阶段标识\n"
                "- read_min/max_score：分数过滤阈值\n"
                "- 其他参数同QuestionCategoryClassifier\n\n"
                "输出参数：\n"
                "- difficulty_score：数值型难度评分（1-10）"
            )
        elif lang == "en":
            return (
                "Evaluates question difficulty level using LLM analysis. "
                "Outputs numerical difficulty score from 1 to 10.\n\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as QuestionCategoryClassifier\n\n"
                "Output Parameters:\n"
                "- difficulty_score: Numerical difficulty rating (1-10)"
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
                used_prompt = self.prompts.question_synthesis_prompt(text)
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