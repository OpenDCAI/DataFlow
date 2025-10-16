import pandas as pd
import numpy as np
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.code.eval.code_text_composition_sample_evaluator import CodeTextCompositionSampleEvaluator

@OPERATOR_REGISTRY.register()
class CodeTextCompositionFilter(OperatorABC):
    """
    CodeTextCompositionFilter filters code samples based on character composition using
    CodeTextCompositionSampleEvaluator scores. It removes binary files, encrypted content,
    and other non-readable text.
    """

    def __init__(self, min_score: float = 1.0, max_score: float = 1.0):
        """
        Initialize the operator with evaluator and thresholds.
        """
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score: {self.min_score} and max_score: {self.max_score}...")
        self.scorer = CodeTextCompositionSampleEvaluator()

    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于CodeTextCompositionSampleEvaluator的得分过滤代码样本，移除二进制文件、加密内容和不可读文本。\n\n"
                "评估指标：\n"
                "- 字母字符比例：普通语言需要>=25%\n"
                "- 字母数字字符比例：汇编语言需要>=25%\n"
                "- 综合字符组成得分：0-1，1表示通过检查\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含'text'和'language'列）\n"
                "- output_key: 输出标签字段名 (默认: 'text_composition_filter_label')\n"
                "- min_score: 最小字符组成得分阈值 (默认: 1.0)\n"
                "- max_score: 最大字符组成得分阈值 (默认: 1.0)\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留字符组成得分在指定范围内的代码样本\n"
                "- 返回包含字符组成得分标签字段名的列表"
            )
        else:
            return (
                "Filter code samples using scores from CodeTextCompositionSampleEvaluator to remove binary files, encrypted content, and non-readable text.\n\n"
                "Evaluation Metrics:\n"
                "- Alphabetic character ratio: Normal languages require >=25%\n"
                "- Alphanumeric character ratio: Assembly languages require >=25%\n"
                "- Comprehensive composition score: 0-1, 1 means passes checks\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires 'text' and 'language' columns)\n"
                "- output_key: Output label field name (default: 'text_composition_filter_label')\n"
                "- min_score: Minimum composition score threshold (default: 1.0)\n"
                "- max_score: Maximum composition score threshold (default: 1.0)\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only code samples with composition scores within specified range\n"
                "- List containing composition score label field name"
            )


    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'text_composition_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        
        # Add scores to dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        # Apply filtering based on CodeTextCompositionScore
        results = np.ones(len(dataframe), dtype=int)
        score_filter = (dataframe["CodeTextCompositionScore"] >= self.min_score) & (dataframe["CodeTextCompositionScore"] <= self.max_score)
        nan_filter = np.isnan(dataframe["CodeTextCompositionScore"])
        metric_filter = score_filter | nan_filter
        results = results & metric_filter.astype(int)
        
        self.logger.debug(f"Filtered by composition score, {np.sum(results)} data remained")
        dataframe[f"{self.output_key}"] = metric_filter.astype(int)
        
        filtered_dataframe = dataframe[results == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [self.output_key]