import pandas as pd
import numpy as np
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.code.eval.code_length_sample_evaluator import CodeLengthSampleEvaluator

@OPERATOR_REGISTRY.register()
class CodeLengthSampleFilter(OperatorABC):
    """
    CodeLengthSampleFilter filters code samples based on length characteristics using
    CodeLengthSampleEvaluator scores. It removes oversized files and poorly formatted code.
    """

    def __init__(self, min_score: float = 1.0, max_score: float = 1.0):
        """
        Initialize the operator with evaluator and thresholds.
        """
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score: {self.min_score} and max_score: {self.max_score}...")
        self.scorer = CodeLengthSampleEvaluator()

    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于CodeLengthSampleEvaluator的得分过滤代码样本，移除超大文件和格式不良的代码。\n\n"
                "评估指标：\n"
                "- 总行数：检查是否超过100,000行\n"
                "- 平均行长：普通语言>100字符，特殊语言>100,000字符\n"
                "- 最大行长：普通语言>1,000字符\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含'lines'和'language'列）\n"
                "- output_key: 输出标签字段名 (默认: 'length_filter_label')\n"
                "- min_score: 最小长度得分阈值 (默认: 1.0)\n"
                "- max_score: 最大长度得分阈值 (默认: 1.0)\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留长度得分在指定范围内的代码样本\n"
                "- 返回包含长度得分标签字段名的列表"
            )
        else:
            return (
                "Filter code samples using scores from CodeLengthSampleEvaluator to remove oversized files and poorly formatted code.\n\n"
                "Evaluation Metrics:\n"
                "- Total lines: Check if exceeds 100,000 lines\n"
                "- Average line length: Normal languages >100 chars, special languages >100,000 chars\n"
                "- Maximum line length: Normal languages >1,000 chars\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires 'lines' and 'language' columns)\n"
                "- output_key: Output label field name (default: 'length_filter_label')\n"
                "- min_score: Minimum length score threshold (default: 1.0)\n"
                "- max_score: Maximum length score threshold (default: 1.0)\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only code samples with length scores within specified range\n"
                "- List containing length score label field name"
            )


    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'length_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        
        # Add scores to dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        # Apply filtering based on CodeLengthScore
        results = np.ones(len(dataframe), dtype=int)
        score_filter = (dataframe["CodeLengthScore"] >= self.min_score) & (dataframe["CodeLengthScore"] <= self.max_score)
        nan_filter = np.isnan(dataframe["CodeLengthScore"])
        metric_filter = score_filter | nan_filter
        results = results & metric_filter.astype(int)
        
        self.logger.debug(f"Filtered by length score, {np.sum(results)} data remained")
        dataframe[f"{self.output_key}"] = metric_filter.astype(int)
        
        filtered_dataframe = dataframe[results == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [self.output_key]