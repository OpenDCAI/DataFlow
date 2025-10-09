import pandas as pd
from typing import List
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeLengthSampleEvaluator(OperatorABC):
    """
    CodeLengthSampleEvaluator evaluates code samples based on length characteristics
    to provide scores for filtering oversized files and poorly formatted code.
    It analyzes total lines, average line length, and maximum line length,
    applying different thresholds for different programming languages.
    """

    # List of languages that require special handling (these languages allow longer line lengths)
    EXCLUDED_LANGS = {
        "HTML", "JSON", "Markdown", "Roff", "Roff Manpage", 
        "SMT", "TeX", "Text", "XML"
    }

    def __init__(self):
        """
        Initialize the operator and set up the logger.
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'CodeLengthScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于代码长度特征评估代码样本，分析总行数、平均行长和最大行长。\n\n"
                "评估指标：\n"
                "- CodeLengthTotalLines: 总行数\n"
                "- CodeLengthAvgLineLength: 平均行长\n"
                "- CodeLengthMaxLineLength: 最大行长\n"
                "- CodeLengthScore: 综合长度得分 (0-1，1表示通过所有长度检查)\n\n"
                "输入要求：需要包含'lines'和'language'列\n\n"
                "输出参数：\n"
                "- CodeLengthTotalLines: 总行数\n"
                "- CodeLengthAvgLineLength: 平均行长\n"
                "- CodeLengthMaxLineLength: 最大行长\n"
                "- CodeLengthScore: 综合长度得分"
            )
        else:
            return (
                "Evaluate code samples based on length characteristics, analyzing total lines, average line length, and maximum line length.\n\n"
                "Evaluation Metrics:\n"
                "- CodeLengthTotalLines: Total number of lines\n"
                "- CodeLengthAvgLineLength: Average line length\n"
                "- CodeLengthMaxLineLength: Maximum line length\n"
                "- CodeLengthScore: Comprehensive length score (0-1, 1 means passes all length checks)\n\n"
                "Input Requirement: Requires 'lines' and 'language' columns\n\n"
                "Output Parameters:\n"
                "- CodeLengthTotalLines: Total number of lines\n"
                "- CodeLengthAvgLineLength: Average line length\n"
                "- CodeLengthMaxLineLength: Maximum line length\n"
                "- CodeLengthScore: Comprehensive length score"
            )

    def _score_func(self, sample):
        """
        Calculate length-based scores for a single code sample.
        
        Args:
            sample: Dictionary containing 'lines' and 'language' keys
            
        Returns:
            Dictionary containing length scores
        """
        lines = sample.get('lines', [])
        language = sample.get('language', '')
        
        # Calculate basic metrics
        n_lines = len(lines)
        avg_len = sum(len(l) for l in lines) / max(1, n_lines)
        max_len = max((len(l) for l in lines), default=0)
        
        # Calculate comprehensive score (0-1)
        score = 1.0
        
        # Check total number of lines
        if n_lines > 100_000:
            score = 0.0
        
        # Apply different rules based on language type
        elif language not in self.EXCLUDED_LANGS:
            if avg_len > 100 or max_len > 1000:
                score = 0.0
        else:
            if max_len > 100_000:
                score = 0.0
        
        return {
            'CodeLengthTotalLines': n_lines,
            'CodeLengthAvgLineLength': avg_len,
            'CodeLengthMaxLineLength': max_len,
            'CodeLengthScore': score
        }

    def eval(self, dataframe, input_key):
        """
        Evaluate length characteristics for all samples in the dataframe.
        
        Args:
            dataframe: Input DataFrame
            input_key: Key containing the sample data
            
        Returns:
            List of score dictionaries
        """
        scores_list = []
        self.logger.info(f"Evaluating {self.score_name}...")
        
        for _, row in dataframe.iterrows():
            sample = row[input_key] if isinstance(row[input_key], dict) else {"lines": row[input_key], "language": "unknown"}
            scores = self._score_func(sample)
            scores_list.append(scores)
        
        self.logger.info("Evaluation complete!")
        return scores_list

    def run(self, storage: DataFlowStorage, input_key: str):
        """
        Execute length evaluation operation.
        
        Args:
            storage: Data storage object
            input_key: Key name for input data
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("CodeLengthScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        storage.write(dataframe)
