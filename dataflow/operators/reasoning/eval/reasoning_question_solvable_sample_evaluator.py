from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.math import MathQuestionEvaluatorPrompt
from typing import Union

import pandas as pd
import json
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@prompt_restrict(
    MathQuestionEvaluatorPrompt
)

@OPERATOR_REGISTRY.register()
class ReasoningQuestionSolvableSampleEvaluator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, prompt_template: Union[MathQuestionEvaluatorPrompt, DIYPromptABC] = None):
        """
        Initialize the ReasoningCategoryDatasetEvaluator with the provided configuration.
        """
        self.logger = get_logger()
        if prompt_template is None:
            prompt_template = MathQuestionEvaluatorPrompt()
        self.prompt_template = prompt_template
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对用户问题进行多级分类（主分类和子分类）。"
                "通过大语言模型对输入问题进行语义分析，输出分类编码结果。\n\n"
                "输入参数：\n"
                "- db_port/db_name/table_name：数据库连接参数（存储模式）\n"
                "- input_file/output_file：文件路径（文件模式）\n"
                "- input_key：输入数据中问题字段的键名\n"
                "- generator_type：模型调用方式（aisuite/request）\n\n"
                "输出参数：\n"
                "- classification_result：包含主分类和子分类的编码结果"
            )
        elif lang == "en":
            return (
                "Performs hierarchical classification (primary and secondary) on user questions. "
                "Utilizes LLM for semantic analysis and outputs category codes.\n\n"
                "Input Parameters:\n"
                "- db_port/db_name/table_name: Database connection params (storage mode)\n"
                "- input_file/output_file: File paths (file mode)\n"
                "- input_key: Key for question field in input data\n"
                "- generator_type: Model invocation method (aisuite/request)\n\n"
                "Output Parameters:\n"
                "- classification_result: Combined category code"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")
        
    def _reformat_prompt(self, dataframe):
        problem = dataframe[self.input_key].tolist()
        system_prompt = self.prompt_template.build_system_prompt()
        prompts = [self.prompt_template.build_prompt(p) for p in problem]

        return system_prompt, prompts

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        """
        Run the question generation process.
        """
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        sys_prompts, user_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        dataframe[f"{output_key}"] = responses
        self.logger.info(f"Generated questions for {output_key}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")