import re
import pandas as pd
import numpy as np
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.code.eval.code_encoded_data_sample_evaluator import CodeEncodedDataSampleEvaluator

@OPERATOR_REGISTRY.register()
class CodeEncodedDataFilter(OperatorABC):
    """
    CodeEncodedDataFilter filters code samples based on encoded data patterns using
    CodeEncodedDataSampleEvaluator scores. It removes binary content and auto-generated code.
    """

    def __init__(self, min_score: float = 1.0, max_score: float = 1.0):
        """
        Initialize the operator with evaluator and thresholds.
        """
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score: {self.min_score} and max_score: {self.max_score}...")
        self.scorer = CodeEncodedDataSampleEvaluator()

    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于CodeEncodedDataSampleEvaluator的得分过滤代码样本，移除二进制内容和自动生成代码。\n\n"
                "评估指标：\n"
                "- Base64编码数据比例：检测连续64+字符的Base64字符串\n"
                "- 十六进制数据比例：检测8+个连续的十六进制对\n"
                "- Unicode转义序列比例：检测8+个连续的\\uXXXX序列\n"
                "- 综合编码数据得分：0-1，1表示通过检查\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含'text'列）\n"
                "- output_key: 输出标签字段名 (默认: 'encoded_data_filter_label')\n"
                "- min_score: 最小编码数据得分阈值 (默认: 1.0)\n"
                "- max_score: 最大编码数据得分阈值 (默认: 1.0)\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留编码数据得分在指定范围内的代码样本\n"
                "- 返回包含编码数据得分标签字段名的列表"
            )
        else:
            return (
                "Filter code samples using scores from CodeEncodedDataSampleEvaluator to remove binary content and auto-generated code.\n\n"
                "Evaluation Metrics:\n"
                "- Base64 encoded data ratio: Detect 64+ consecutive Base64 characters\n"
                "- Hexadecimal data ratio: Detect 8+ consecutive hex pairs\n"
                "- Unicode escape sequence ratio: Detect 8+ consecutive \\uXXXX sequences\n"
                "- Comprehensive encoded data score: 0-1, 1 means passes checks\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires 'text' column)\n"
                "- output_key: Output label field name (default: 'encoded_data_filter_label')\n"
                "- min_score: Minimum encoded data score threshold (default: 1.0)\n"
                "- max_score: Maximum encoded data score threshold (default: 1.0)\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only code samples with encoded data scores within specified range\n"
                "- List containing encoded data score label field name"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = [self.input_text_key]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"DataPatternFilter missing required columns: {missing}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str,
        output_key: str = "encoded_data_filter_label"
    ) -> List[str]:
        """
        Execute data pattern detection and filtering using evaluator scores.

        Args:
            storage: Data storage object
            input_key: Field name containing code text
            output_key: Key name for output label

        Returns:
            List[str]: List containing output key name
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        
        # Add scores to dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        # Apply filtering based on CodeEncodedDataScore
        results = np.ones(len(dataframe), dtype=int)
        score_filter = (dataframe["CodeEncodedDataScore"] >= self.min_score) & (dataframe["CodeEncodedDataScore"] <= self.max_score)
        nan_filter = np.isnan(dataframe["CodeEncodedDataScore"])
        metric_filter = score_filter | nan_filter
        results = results & metric_filter.astype(int)
        
        self.logger.debug(f"Filtered by encoded data score, {np.sum(results)} data remained")
        dataframe[f"{self.output_key}"] = metric_filter.astype(int)
        
        filtered_dataframe = dataframe[results == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [self.output_key]