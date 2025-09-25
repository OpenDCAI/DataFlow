import pandas as pd
import re
import math
from typing import List, Dict, Any
from collections import Counter
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeDocumentQualitySampleEvaluator(OperatorABC):
    """
    CodeDocumentQualitySampleEvaluator evaluates code samples based on comprehensive
    document-level quality metrics to provide scores for filtering low-quality content.
    """

    def __init__(self, thresholds: Dict[str, Any] = None):
        """
        Initialize the operator and set up thresholds.
        
        Args:
            thresholds: Optional thresholds dictionary to override defaults
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'CodeDocumentQualityScore'
        
        # Default thresholds for each rule
        default_thresholds = {
            'min_num_chars': 1,
            'max_num_chars': 100000,
            'min_num_words': 1,
            'max_num_words': 100000,
            'max_frac_duplicate_lines': 1.0,
            'max_frac_duplicate_2gram': 1.0,
            'max_frac_duplicate_3gram': 1.0,
            'max_frac_duplicate_4gram': 1.0,
            'max_frac_duplicate_5gram': 1.0,
            'max_frac_duplicate_6gram': 1.0,
            'max_frac_duplicate_7gram': 1.0,
            'max_frac_duplicate_8gram': 1.0,
            'max_frac_duplicate_9gram': 1.0,
            'max_frac_duplicate_10gram': 1.0,
            'max_frac_curly_bracket': 1.0,
            'max_frac_all_caps_words': 1.0,
            'min_entropy_unigram': 0.0,
        }
        
        # Merge with provided thresholds
        self.thresholds = default_thresholds.copy()
        if thresholds:
            self.thresholds.update(thresholds)
            
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "基于综合文档级质量指标评估代码样本，包括内容长度、重复模式、字符组成和文本熵值。\n\n"
                "评估指标：\n"
                "- CodeDocumentQualityCharCount: 字符数\n"
                "- CodeDocumentQualityWordCount: 词数\n"
                "- CodeDocumentQualityDuplicateLinesRatio: 重复行比例\n"
                "- CodeDocumentQualityDuplicateNgramRatio: n-gram重复比例\n"
                "- CodeDocumentQualityCurlyBracketRatio: 花括号比例\n"
                "- CodeDocumentQualityAllCapsRatio: 全大写单词比例\n"
                "- CodeDocumentQualityEntropy: 单字符熵值\n"
                "- CodeDocumentQualityScore: 综合文档质量得分 (0-1，1表示通过所有质量检查)\n\n"
                "输入要求：需要包含'text'、'filename'、'language'列\n\n"
                "输出参数：\n"
                "- 各种质量指标的数值\n"
                "- CodeDocumentQualityScore: 综合文档质量得分"
            )
        else:
            return (
                "Evaluate code samples based on comprehensive document-level quality metrics including content length, repetition patterns, character composition, and text entropy.\n\n"
                "Evaluation Metrics:\n"
                "- CodeDocumentQualityCharCount: Character count\n"
                "- CodeDocumentQualityWordCount: Word count\n"
                "- CodeDocumentQualityDuplicateLinesRatio: Duplicate lines ratio\n"
                "- CodeDocumentQualityDuplicateNgramRatio: n-gram repetition ratio\n"
                "- CodeDocumentQualityCurlyBracketRatio: Curly bracket ratio\n"
                "- CodeDocumentQualityAllCapsRatio: All-caps words ratio\n"
                "- CodeDocumentQualityEntropy: Unigram entropy\n"
                "- CodeDocumentQualityScore: Comprehensive document quality score (0-1, 1 means passes all quality checks)\n\n"
                "Input Requirement: Requires 'text', 'filename', 'language' columns\n\n"
                "Output Parameters:\n"
                "- Various quality metric values\n"
                "- CodeDocumentQualityScore: Comprehensive document quality score"
            )

    def _frac_duplicate_ngrams_n(self, text: str, n: int) -> float:
        """Calculate the fraction of duplicate n-grams in the text"""
        words = re.findall(r'\b\w+\b', text)
        if len(words) < n:
            return 0.0
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        ngram2count = Counter(ngrams)
        count = sum([v for v in ngram2count.values() if v != 1])
        total = sum([v for v in ngram2count.values()])
        return count / total if total else 0.0

    def _num_chars(self, text: str) -> int:
        return len(text)

    def _num_words(self, text: str) -> int:
        return len(re.findall(r'\w+', text))

    def _frac_duplicate_lines(self, lines: List[str]) -> float:
        if not lines:
            return 0.0
        line2count = Counter([l.strip() for l in lines if l.strip()])
        count = sum([v for v in line2count.values() if v != 1])
        total = sum([v for v in line2count.values()])
        return count / total if total else 0.0

    def _frac_curly_bracket(self, text: str) -> float:
        total = len(text)
        if total == 0:
            return 0.0
        count = text.count('{') + text.count('}')
        return count / total

    def _frac_all_caps_words(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        all_caps = [w for w in words if w.isupper() and len(w) > 1]
        return len(all_caps) / len(words)

    def _entropy_unigram(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        freq = Counter(words)
        total = sum(freq.values())
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
        return entropy

    def _score_func(self, sample):
        """
        Calculate document quality scores for a single code sample.
        
        Args:
            sample: Dictionary containing 'text', 'filename', 'language' keys
            
        Returns:
            Dictionary containing document quality scores
        """
        text = sample.get('text', sample.get('content', ''))
        filename = sample.get('filename', '')
        lang = sample.get('language', 'en')
        lines = text.splitlines()
        
        # Use instance thresholds
        thresh = self.thresholds
        
        # Calculate individual metrics
        num_chars = self._num_chars(text)
        num_words = self._num_words(text)
        frac_dup_lines = self._frac_duplicate_lines(lines)
        frac_curly = self._frac_curly_bracket(text)
        frac_caps = self._frac_all_caps_words(text)
        entropy = self._entropy_unigram(text)
        
        # Calculate n-gram ratios
        ngram_ratios = {}
        for n in range(2, 11):
            key = f'max_frac_duplicate_{n}gram'
            if key in thresh:
                ngram_ratios[f'duplicate_{n}gram_ratio'] = self._frac_duplicate_ngrams_n(text, n)
        
        # Calculate comprehensive score (0-1)
        score = 1.0
        
        # Apply all quality rules
        if num_chars < thresh['min_num_chars'] or num_chars > thresh['max_num_chars']:
            score = 0.0
        elif num_words < thresh['min_num_words'] or num_words > thresh['max_num_words']:
            score = 0.0
        elif frac_dup_lines > thresh['max_frac_duplicate_lines']:
            score = 0.0
        elif frac_curly > thresh['max_frac_curly_bracket']:
            score = 0.0
        elif frac_caps > thresh['max_frac_all_caps_words']:
            score = 0.0
        elif entropy < thresh['min_entropy_unigram']:
            score = 0.0
        else:
            # Check n-gram ratios
            for n in range(2, 11):
                key = f'max_frac_duplicate_{n}gram'
                if key in thresh:
                    frac_dup_ngram = self._frac_duplicate_ngrams_n(text, n)
                    if frac_dup_ngram > thresh[key]:
                        score = 0.0
                        break
        
        # Prepare result dictionary
        result = {
            'CodeDocumentQualityCharCount': num_chars,
            'CodeDocumentQualityWordCount': num_words,
            'CodeDocumentQualityDuplicateLinesRatio': frac_dup_lines,
            'CodeDocumentQualityCurlyBracketRatio': frac_curly,
            'CodeDocumentQualityAllCapsRatio': frac_caps,
            'CodeDocumentQualityEntropy': entropy,
            'CodeDocumentQualityScore': score
        }
        
        # Add n-gram ratios
        result.update({f'CodeDocumentQuality{k.title()}Ratio': v for k, v in ngram_ratios.items()})
        
        return result

    def eval(self, dataframe, input_key):
        """
        Evaluate document quality for all samples in the dataframe.
        
        Args:
            dataframe: Input DataFrame
            input_key: Key containing the sample data
            
        Returns:
            List of score dictionaries
        """
        scores_list = []
        self.logger.info(f"Evaluating {self.score_name}...")
        
        for _, row in dataframe.iterrows():
            sample = row[input_key] if isinstance(row[input_key], dict) else {"text": row[input_key], "filename": "", "language": "unknown"}
            scores = self._score_func(sample)
            scores_list.append(scores)
        
        self.logger.info("Evaluation complete!")
        return scores_list

    def run(self, storage: DataFlowStorage, input_key: str):
        """
        Execute document quality evaluation operation.
        
        Args:
            storage: Data storage object
            input_key: Key name for input data
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("CodeDocumentQualityScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        
        storage.write(dataframe)
