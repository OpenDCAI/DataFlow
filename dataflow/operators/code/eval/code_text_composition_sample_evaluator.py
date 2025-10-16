import pandas as pd
from typing import Dict
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeTextCompositionSampleEvaluator(OperatorABC):
    """
    CodeTextCompositionSampleEvaluator evaluates code samples based on character composition
    to provide scores for filtering binary files, encrypted content, and other non-readable text.
    It analyzes the ratio of alphabetic and alphanumeric characters to ensure readable content.
    """

    # List of languages that require special handling
    SPECIAL_LANGS = {"Motorola 68K Assembly", "WebAssembly"}

    def __init__(self):
        """
        Initialize the operator and set up the logger.
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'CodeTextCompositionScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于字符组成评估代码样本，分析字母字符和字母数字字符的比例。\n\n"
                "评估指标：\n"
                "- CodeTextCompositionAlphaRatio: 字母字符比例\n"
                "- CodeTextCompositionAlnumRatio: 字母数字字符比例\n"
                "- CodeTextCompositionScore: 综合字符组成得分 (0-1，1表示通过字符组成检查)\n\n"
                "输入要求：需要包含'text'和'language'列\n\n"
                "输出参数：\n"
                "- CodeTextCompositionAlphaRatio: 字母字符比例\n"
                "- CodeTextCompositionAlnumRatio: 字母数字字符比例\n"
                "- CodeTextCompositionScore: 综合字符组成得分"
            )
        else:
            return (
                "Evaluate code samples based on character composition, analyzing ratios of alphabetic and alphanumeric characters.\n\n"
                "Evaluation Metrics:\n"
                "- CodeTextCompositionAlphaRatio: Alphabetic character ratio\n"
                "- CodeTextCompositionAlnumRatio: Alphanumeric character ratio\n"
                "- CodeTextCompositionScore: Comprehensive composition score (0-1, 1 means passes composition checks)\n\n"
                "Input Requirement: Requires 'text' and 'language' columns\n\n"
                "Output Parameters:\n"
                "- CodeTextCompositionAlphaRatio: Alphabetic character ratio\n"
                "- CodeTextCompositionAlnumRatio: Alphanumeric character ratio\n"
                "- CodeTextCompositionScore: Comprehensive composition score"
            )

    def _score_func(self, sample):
        """
        Calculate composition-based scores for a single code sample.
        
        Args:
            sample: Dictionary containing 'text' and 'language' keys
            
        Returns:
            Dictionary containing composition scores
        """
        text = sample.get('text', '')
        language = sample.get('language', '')
        
        # Calculate character ratios
        alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
        alnum_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
        
        # Calculate comprehensive score (0-1)
        score = 1.0
        
        if language in self.SPECIAL_LANGS:
            # For assembly languages, check alphanumeric character ratio
            if alnum_ratio < 0.25:
                score = 0.0
        else:
            # For normal languages, check alphabetic character ratio
            if alpha_ratio < 0.25:
                score = 0.0
        
        return {
            'CodeTextCompositionAlphaRatio': alpha_ratio,
            'CodeTextCompositionAlnumRatio': alnum_ratio,
            'CodeTextCompositionScore': score
        }

    def eval(self, dataframe, input_key):
        """
        Evaluate character composition for all samples in the dataframe.
        
        Args:
            dataframe: Input DataFrame
            input_key: Key containing the sample data
            
        Returns:
            List of score dictionaries
        """
        scores_list = []
        self.logger.info(f"Evaluating {self.score_name}...")
        
        for _, row in dataframe.iterrows():
            sample = row[input_key] if isinstance(row[input_key], dict) else {"text": row[input_key], "language": "unknown"}
            scores = self._score_func(sample)
            scores_list.append(scores)
        
        self.logger.info("Evaluation complete!")
        return scores_list

    def run(self, storage: DataFlowStorage, input_key: str):
        """
        Execute character composition evaluation operation.
        
        Args:
            storage: Data storage object
            input_key: Key name for input data
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("CodeTextCompositionScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        storage.write(dataframe)
