import re
import pandas as pd
from typing import Dict
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeEncodedDataSampleEvaluator(OperatorABC):
    """
    CodeEncodedDataSampleEvaluator evaluates code samples based on encoded data patterns
    to provide scores for filtering binary content and auto-generated code.
    """

    def __init__(self):
        """
        Initialize the operator and set up the logger.
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'CodeEncodedDataScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于编码数据模式评估代码样本，检测Base64、十六进制和Unicode转义序列。\n\n"
                "评估指标：\n"
                "- CodeEncodedDataBase64Ratio: Base64编码数据比例\n"
                "- CodeEncodedDataHexRatio: 十六进制数据比例\n"
                "- CodeEncodedDataUnicodeRatio: Unicode转义序列比例\n"
                "- CodeEncodedDataScore: 综合编码数据得分 (0-1，1表示通过编码数据检查)\n\n"
                "输入要求：需要包含'text'列\n\n"
                "输出参数：\n"
                "- CodeEncodedDataBase64Ratio: Base64编码数据比例\n"
                "- CodeEncodedDataHexRatio: 十六进制数据比例\n"
                "- CodeEncodedDataUnicodeRatio: Unicode转义序列比例\n"
                "- CodeEncodedDataScore: 综合编码数据得分"
            )
        else:
            return (
                "Evaluate code samples based on encoded data patterns, detecting Base64, hexadecimal, and Unicode escape sequences.\n\n"
                "Evaluation Metrics:\n"
                "- CodeEncodedDataBase64Ratio: Base64 encoded data ratio\n"
                "- CodeEncodedDataHexRatio: Hexadecimal data ratio\n"
                "- CodeEncodedDataUnicodeRatio: Unicode escape sequence ratio\n"
                "- CodeEncodedDataScore: Comprehensive encoded data score (0-1, 1 means passes encoded data checks)\n\n"
                "Input Requirement: Requires 'text' column\n\n"
                "Output Parameters:\n"
                "- CodeEncodedDataBase64Ratio: Base64 encoded data ratio\n"
                "- CodeEncodedDataHexRatio: Hexadecimal data ratio\n"
                "- CodeEncodedDataUnicodeRatio: Unicode escape sequence ratio\n"
                "- CodeEncodedDataScore: Comprehensive encoded data score"
            )

    def _score_func(self, sample):
        """
        Calculate encoded data pattern scores for a single code sample.
        
        Args:
            sample: Dictionary containing 'text' key or string
            
        Returns:
            Dictionary containing encoded data scores
        """
        if isinstance(sample, str):
            text = sample
        else:
            text = sample.get('text', '')
        
        # Compile regular expressions for different data patterns
        patterns = [
            re.compile(r"[a-zA-Z0-9+/=\n]{64,}"),  # Base64
            re.compile(r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}"),  # Hex
            re.compile(r"(?:\\u[0-9a-fA-F]{4}){8,}")  # Unicode
        ]
        
        # Calculate ratios for each pattern
        base64_ratio = 0.0
        hex_ratio = 0.0
        unicode_ratio = 0.0
        
        for i, pattern in enumerate(patterns):
            total_match_len = 0
            for match in pattern.finditer(text):
                match_len = len(match.group())
                total_match_len += match_len
            
            ratio = total_match_len / max(1, len(text))
            
            if i == 0:  # Base64
                base64_ratio = ratio
            elif i == 1:  # Hex
                hex_ratio = ratio
            else:  # Unicode
                unicode_ratio = ratio
        
        # Calculate comprehensive score (0-1)
        score = 1.0
        
        # Check if any pattern exceeds thresholds
        if base64_ratio > 0.5 or hex_ratio > 0.5 or unicode_ratio > 0.5:
            score = 0.0
        
        return {
            'CodeEncodedDataBase64Ratio': base64_ratio,
            'CodeEncodedDataHexRatio': hex_ratio,
            'CodeEncodedDataUnicodeRatio': unicode_ratio,
            'CodeEncodedDataScore': score
        }

    def eval(self, dataframe, input_key):
        """
        Evaluate encoded data patterns for all samples in the dataframe.
        
        Args:
            dataframe: Input DataFrame
            input_key: Key containing the sample data
            
        Returns:
            List of score dictionaries
        """
        scores_list = []
        self.logger.info(f"Evaluating {self.score_name}...")
        
        for _, row in dataframe.iterrows():
            text = row[input_key] if isinstance(row[input_key], str) else row[input_key].get('text', '')
            scores = self._score_func(text)
            scores_list.append(scores)
        
        self.logger.info("Evaluation complete!")
        return scores_list

    def run(self, storage: DataFlowStorage, input_key: str):
        """
        Execute encoded data pattern evaluation operation.
        
        Args:
            storage: Data storage object
            input_key: Key name for input data
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("CodeEncodedDataScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        storage.write(dataframe)
