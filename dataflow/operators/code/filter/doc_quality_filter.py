import pandas as pd
from typing import List, Dict, Any

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import re
from collections import Counter

@OPERATOR_REGISTRY.register()
class DocQualityFilter(OperatorABC):
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

    """
    DocQualityFilter applies a set of document-level quality filtering rules to a dataset.
    These include natural language, general code, and document repetition/statistics rules.
    Each rule is implemented as a method and can be configured via thresholds.
    
    The filter helps remove low-quality documents by checking various quality metrics
    such as character/word counts, duplicate content ratios, and text composition.
    """

    def __init__(self):
        """Initialize the operator and set up default thresholds"""
        self.logger = get_logger()
        # Default thresholds for each rule (can be overridden via run params)
        self.thresholds = {
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

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "该算子对文档级别的质量信号进行过滤，包括字符数、词数、重复行比例、n-gram重复、括号比例等。"
            )
        else:
            return (
                "This operator applies document-level quality filtering rules, including character/word count, duplicate line ratio, n-gram repetition, bracket ratio, etc."
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

    def run(self, storage: DataFlowStorage, thresholds: Dict[str, Any] = None) -> List[str]:
        """
        Applies document-level quality filtering rules to the input dataframe.
        """
        self.logger.info("Running DocQualityFilter...")
        if thresholds is not None:
            self.thresholds.update(thresholds)
        df = storage.read("dataframe")
        if df.empty:
            self.logger.warning("Input dataframe for DocQualityFilter is empty. Skipping.")
            storage.write(df)
            return []
        original_count = len(df)
        # Apply all rules row-wise
        filtered_df = df[df.apply(lambda row: self._apply_rules(row, self.thresholds), axis=1)]
        filtered_count = len(filtered_df)
        self.logger.info(f"Filtering complete. Kept {filtered_count} / {original_count} rows.")
        output_file = storage.write(filtered_df)
        self.logger.success(f"DocQualityFilter finished. Filtered data saved to {output_file}")
        return []
