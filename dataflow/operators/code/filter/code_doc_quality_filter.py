import pandas as pd
import numpy as np
from typing import List, Dict, Any

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.operators.code.eval.code_document_quality_sample_evaluator import CodeDocumentQualitySampleEvaluator

import re
from collections import Counter

@OPERATOR_REGISTRY.register()
class CodeDocumentQualityFilter(OperatorABC):
    """
    CodeDocumentQualityFilter applies comprehensive document-level quality filtering
    rules using CodeDocumentQualitySampleEvaluator scores to remove low-quality code and text samples.
    """

    def __init__(self, min_score: float = 1.0, max_score: float = 1.0, thresholds: Dict[str, Any] = None):
        """
        Initialize the operator with evaluator and thresholds.
        
        Args:
            min_score: Minimum document quality score threshold
            max_score: Maximum document quality score threshold
            thresholds: Optional thresholds dictionary to override default thresholds
        """
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score: {self.min_score} and max_score: {self.max_score}...")
        self.scorer = CodeDocumentQualitySampleEvaluator(thresholds)   

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "基于CodeDocumentQualitySampleEvaluator的得分应用综合文档级质量过滤规则，移除低质量代码和文本样本。\n\n"
                "评估指标：\n"
                "- 内容长度：字符数、词数、行数范围检查\n"
                "- 重复模式：重复行比例、2-10gram重复比例\n"
                "- 字符组成：花括号比例、全大写单词比例\n"
                "- 文本熵值：单字符熵值检查\n"
                "- 综合文档质量得分：0-1，1表示通过所有质量检查\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含'text'、'filename'、'language'列）\n"
                "- output_key: 输出标签字段名 (默认: 'doc_quality_filter_label')\n"
                "- min_score: 最小文档质量得分阈值 (默认: 1.0)\n"
                "- max_score: 最大文档质量得分阈值 (默认: 1.0)\n"
                "- thresholds: 可选的阈值字典，用于覆盖默认阈值\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留文档质量得分在指定范围内的样本\n"
                "- 返回包含文档质量得分标签字段名的列表"
            )
        else:
            return (
                "Apply comprehensive document-level quality filtering rules using scores from CodeDocumentQualitySampleEvaluator to remove low-quality code and text samples.\n\n"
                "Evaluation Metrics:\n"
                "- Content length: character/word/line count range checks\n"
                "- Repetition patterns: duplicate line ratio, 2-10gram repetition ratios\n"
                "- Character composition: curly bracket ratio, all-caps word ratio\n"
                "- Text entropy: unigram entropy checks\n"
                "- Comprehensive document quality score: 0-1, 1 means passes all quality checks\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires 'text', 'filename', 'language' columns)\n"
                "- output_key: Output label field name (default: 'doc_quality_filter_label')\n"
                "- min_score: Minimum document quality score threshold (default: 1.0)\n"
                "- max_score: Maximum document quality score threshold (default: 1.0)\n"
                "- thresholds: Optional thresholds dictionary to override default thresholds\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with document quality scores within specified range\n"
                "- List containing document quality score label field name"
            )

    def _num_chars(self, text: str) -> int:
        return len(text)

    def _num_words(self, text: str) -> int:
        return len(re.findall(r'\w+', text))

    def _num_lines(self, text: str) -> int:
        return len(text.splitlines())

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
        import math
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
        return entropy

    def _frac_duplicate_ngrams(self, text: str, n: int = 5) -> float:
        words = re.findall(r'\b\w+\b', text)
        if len(words) < n:
            return 0.0
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        ngram2count = Counter(ngrams)
        count = sum([v for v in ngram2count.values() if v != 1])
        total = sum([v for v in ngram2count.values()])
        return count / total if total else 0.0

    def _num_sentences(self, text: str) -> int:
        SENT_PATTERN = re.compile(r'\b[^.!?。！？؟]+[.!?。！？؟]*', flags=re.UNICODE)
        return len(SENT_PATTERN.findall(text))

    def _num_lines(self, text: str) -> int:
        return len(text.splitlines())

    def _mean_word_length(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _frac_full_bracket(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        count = sum(1 for w in words if w in ('【', '】'))
        return count / len(words)

    def _frac_lines_end_with_readmore(self, text: str) -> float:
        ellipsis = ("...", "…", '全文', '详情', '详细', '更多', 'المزيد', 'تفاصيل', 'اقرأ المزيد', 'もっと', '詳細', 'もっと読む')
        lines = text.splitlines()
        if not lines:
            return 0.0
        total_ellipsis_lines = sum(
            any(l.rstrip().rstrip(']】)>》').endswith(e) for e in ellipsis)
            for l in lines
        )
        return total_ellipsis_lines / len(lines)

    def _frac_lines_start_with_bullet(self, text: str) -> float:
        # Common bullet symbols
        bullets = ("-", "*", "•", "·", "●", "▪", "‣", "⁃", "◦", "‧", "﹒", "・", "∙", "‣", "➤", "➢", "➣", "➥", "➦", "➧", "➨", "➩", "➪", "➫", "➬", "➭", "➮", "➯", "➱", "➲", "➳", "➵", "➸", "➺", "➻", "➼", "➽", "➾")
        lines = text.splitlines()
        if not lines:
            return 0.0
        total_bullet_lines = sum(any(l.lstrip().startswith(b) for b in bullets) for l in lines)
        return total_bullet_lines / len(lines)

    def _frac_words_unique(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _frac_replacement_symbols(self, text: str) -> float:
        total = len(text)
        if total == 0:
            return 0.0
        return text.count('�') / total

    def _mean_sentence_length(self, text: str) -> float:
        sentences = re.split(r'\.|\?|\!|\n', text)
        if not sentences:
            return 0.0
        return sum(len(s) for s in sentences) / len(sentences)

    def _frac_chars_url_html(self, text: str) -> float:
        total = len(text)
        if total == 0:
            return 0.0
        link_pattern = r'\(https?://\S+\)'
        html_tag_pattern = r'<.*?>'
        link_list = re.findall(link_pattern, text)
        html_tag_list = re.findall(html_tag_pattern, text)
        url_char_num = sum(len(link) for link in link_list)
        html_tag_char_num = sum(len(tag) for tag in html_tag_list)
        return (url_char_num + html_tag_char_num) / total

    def _frac_chars_alphabet(self, text: str, lang: str = 'en') -> float:
        if not text or lang != 'en':
            return 0.0
        return sum(c.isalpha() for c in text) / len(text)

    def _frac_chars_digital(self, text: str) -> float:
        if not text:
            return 0.0
        return sum(c.isdigit() for c in text) / len(text)

    def _frac_chars_whitespace(self, text: str) -> float:
        if not text:
            return 0.0
        return len(re.findall(r'\s', text)) / len(text)

    def _frac_chars_hex_words(self, text: str) -> float:
        total = len(text)
        if total == 0:
            return 0.0
        count = sum(len(e) for e in re.findall(r'\b0[xX][0-9a-fA-F]+\b', text))
        return count / total

    def _is_code_related_filename(self, filename: str) -> bool:
        related = ["readme", "notes", "todo", "description", "cmakelists"]
        name = filename.split('.')[0].lower()
        return (
            "requirement" in name or name in related or name == "read.me"
        )

    def _apply_rules(self, row: pd.Series, thresholds: Dict[str, Any]) -> bool:
        text = row.get('text', row.get('content', ''))
        filename = row.get('filename', '')
        lang = row.get('language', 'en')
        lines = text.splitlines()
        # Rule 1: min/max chars
        num_chars = self._num_chars(text)
        if num_chars < thresholds['min_num_chars'] or num_chars > thresholds['max_num_chars']:
            return False
        # Rule 2: min/max words
        num_words = self._num_words(text)
        if num_words < thresholds['min_num_words'] or num_words > thresholds['max_num_words']:
            return False
        # Rule 3: duplicate lines
        frac_dup_lines = self._frac_duplicate_lines(lines)
        if frac_dup_lines > thresholds['max_frac_duplicate_lines']:
            return False
        # Rule 4: duplicate n-grams (2~10)
        for n in range(2, 11):
            key = f'max_frac_duplicate_{n}gram'
            if key in thresholds:
                frac_dup_ngram = self._frac_duplicate_ngrams_n(text, n=n)
                if frac_dup_ngram > thresholds[key]:
                    return False
        # Rule 5: curly bracket ratio
        frac_curly = self._frac_curly_bracket(text)
        if frac_curly > thresholds['max_frac_curly_bracket']:
            return False
        # Rule 6: all caps words
        frac_caps = self._frac_all_caps_words(text)
        if frac_caps > thresholds['max_frac_all_caps_words']:
            return False
        # Rule 7: unigram entropy
        entropy = self._entropy_unigram(text)
        if entropy < thresholds['min_entropy_unigram']:
            return False
        # 其它规则同前
        return True

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = "doc_quality_filter_label") -> List[str]:
        """
        Applies document-level quality filtering rules using evaluator scores.
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
        
        # Apply filtering based on CodeDocumentQualityScore
        results = np.ones(len(dataframe), dtype=int)
        score_filter = (dataframe["CodeDocumentQualityScore"] >= self.min_score) & (dataframe["CodeDocumentQualityScore"] <= self.max_score)
        nan_filter = np.isnan(dataframe["CodeDocumentQualityScore"])
        metric_filter = score_filter | nan_filter
        results = results & metric_filter.astype(int)
        
        self.logger.debug(f"Filtered by document quality score, {np.sum(results)} data remained")
        dataframe[f"{self.output_key}"] = metric_filter.astype(int)
        
        filtered_dataframe = dataframe[results == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [self.output_key]
