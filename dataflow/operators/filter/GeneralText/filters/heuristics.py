from dataflow.core import OperatorABC
from typing import Callable, Tuple
import numpy as np
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.cli_funcs.paths import DataFlowPath
from tqdm import tqdm
import re

@OPERATOR_REGISTRY.register()
class ColonEndFilter(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本是否以冒号结尾，过滤掉以冒号结尾的文本" if lang == "zh" else "Check if the text ends with a colon and filter out texts that end with a colon."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = None):
        self.input_key = input_key
        self.output_key = output_key or f"{self.__class__.__name__.lower()}_label"
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")
        colon_end_checks = []
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                colon_end_checks.append(not text.endswith(':'))
            else:
                colon_end_checks.append(0)
        colon_end_checks = np.array(colon_end_checks, dtype=int)
        dataframe[self.output_key] = colon_end_checks
        filtered_dataframe = dataframe[colon_end_checks == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]

@OPERATOR_REGISTRY.register()
class WordNumberFilter(OperatorABC):

    def __init__(self, min_words: int=20, max_words: int=100000):
        self.logger = get_logger()
        self.min_words = min_words
        self.max_words = max_words
        self.logger.info(f"Initializing {self.__class__.__name__} with min_words = {self.min_words}, max_words = {self.max_words}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的单词数量是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the number of words in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='word_number_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")
        word_counts = []
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                normalized_words = tuple(text.split())
                num_normalized_words = len(normalized_words)
                word_counts.append(num_normalized_words)
            else:
                word_counts.append(0)
        word_counts = np.array(word_counts)
        metric_filter = (self.min_words <= word_counts) & (word_counts < self.max_words)
        dataframe[self.output_key] = word_counts
        filtered_dataframe = dataframe[metric_filter]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]


@OPERATOR_REGISTRY.register()
class SentenceNumberFilter(OperatorABC):

    def __init__(self, min_sentences: int=3, max_sentences: int=7500):
        self.logger = get_logger()
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.logger.info(f"Initializing {self.__class__.__name__} with min_sentences = {self.min_sentences}, max_sentences = {self.max_sentences}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的句子数量是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the number of sentences in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'sentence_number_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_check = []
        SENT_PATTERN = re.compile(r'\b[^.!?\n]+[.!?]*', flags=re.UNICODE)

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                num_sentence = len(SENT_PATTERN.findall(text))
                valid_check.append(self.min_sentences <= num_sentence <= self.max_sentences)
            else:
                valid_check.append(0)

        valid_check = np.array(valid_check, dtype=int)
        dataframe[self.output_key] = valid_check
        filtered_dataframe = dataframe[valid_check == 1]
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]



class TextSlice:
    # A slice of text from a document.
    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

def split_paragraphs(
        text: str, normalizer: Callable[[str], str], remove_empty: bool = True
) -> Tuple[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.
    """
    text_slices = tuple(
        TextSlice(normalizer(text[match.start():match.end()]), match.start(), match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    )

    if remove_empty is True:
        text_slices = tuple(
            text_slice for text_slice in text_slices if text_slice.text.strip()
        )

    return text_slices

def normalize(
        text: str,
        remove_punct: bool = True,
        lowercase: bool = True,
        nfd_unicode: bool = True,
        white_space: bool = True
) -> str:
    import string
    import unicodedata
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # lowercase
    if lowercase:
        text = text.lower()

    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)

    return text

@OPERATOR_REGISTRY.register()
class LineEndWithEllipsisFilter(OperatorABC):

    def __init__(self, threshold: float=0.3):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本行是否以省略号结尾，过滤掉以省略号结尾的文本行" if lang == "zh" else "Check if the lines in the text end with ellipsis and filter out lines that end with ellipsis."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'line_end_with_ellipsis_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        ellipsis_checks = []
        ellipsis = ["...", "…"]

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                raw_lines = split_paragraphs(text=text, normalizer=lambda x: x, remove_empty=True)
                num_lines = len(raw_lines)

                if num_lines == 0:
                    ellipsis_checks.append(False)
                    continue

                num_occurrences = sum([line.text.rstrip().endswith(tuple(ellipsis)) for line in raw_lines])
                ratio = num_occurrences / num_lines
                ellipsis_checks.append(ratio < self.threshold)
            else:
                ellipsis_checks.append(False)

        ellipsis_checks = np.array(ellipsis_checks, dtype=int)
        dataframe[self.output_key] = ellipsis_checks
        filtered_dataframe = dataframe[ellipsis_checks == 1]
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

    
@OPERATOR_REGISTRY.register()
class ContentNullFilter(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本内容是否为空，过滤掉空文本" if lang == "zh" else "Check if the text content is empty and filter out empty texts."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='content_null_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        null_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            null_checks.append(text is not None and text.strip() != '')

        null_checks = np.array(null_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = null_checks
        filtered_dataframe = dataframe[null_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class SymbolWordRatioFilter(OperatorABC):

    def __init__(self, threshold: float=0.4):
        self.logger = get_logger()
        self.threshold = threshold
        self.symbol = ["#", "...", "…"]
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的符号与单词的比例是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the ratio of symbols to words in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='symbol_word_ratio_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                raw_words = tuple(WordPunctTokenizer().tokenize(text))
                num_words = len(raw_words)
                num_symbols = float(sum(text.count(symbol) for symbol in self.symbol))

                if num_words == 0:
                    valid_checks.append(False)
                    continue

                ratio = num_symbols / num_words
                valid_checks.append(ratio < self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class AlphaWordsFilter(OperatorABC):

    def __init__(self, threshold: float, use_tokenizer: bool):
        import nltk
        nltk.download('punkt_tab')
        self.logger = get_logger()
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中包含字母字符的单词比例是否大于0.6，过滤掉不符合条件的文本" if lang == "zh" else "Check whether the ratio of words that contain at least one alphabetic character is greater than 0.6 and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='alpha_words_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []
        
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if self.use_tokenizer:
                words = word_tokenize(text)
            else:
                words = text.split()
            alpha_count = sum(1 for word in words if re.search(r'[a-zA-Z]', word))
            word_count = len(words)
            if word_count > 0:
                ratio = alpha_count / word_count
                valid_checks.append(ratio > self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)
        dataframe[self.output_key] = valid_checks
        # Filter the dataframe based on the result
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class HtmlEntityFilter(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否包含HTML实体，过滤掉包含HTML实体的文本" if lang == "zh" else "Check if the text contains HTML entities and filter out texts that contain HTML entities."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='html_entity_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        # Define the list of HTML entities
        html_entity = ["nbsp", "lt", "gt", "amp", "quot", "apos", "hellip", "ndash", "mdash", "lsquo", "rsquo", "ldquo", "rdquo"]
        full_entities_1 = [f"&{entity}；" for entity in html_entity]
        full_entities_2 = [f"&{entity};" for entity in html_entity]
        full_entities_3 = [f"＆{entity}；" for entity in html_entity]
        full_entities_4 = [f"＆{entity};" for entity in html_entity]
        half_entities = [f"＆{entity}" for entity in html_entity] + [f"&{entity}" for entity in html_entity]
        all_entities = full_entities_1 + full_entities_2 + full_entities_3 + full_entities_4 + half_entities

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                has_html_entity = any(entity in text for entity in all_entities)
                valid_checks.append(not has_html_entity)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class IDCardFilter(OperatorABC):

    def __init__(self, threshold:int=3):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否包含身份证相关内容，过滤掉包含身份证相关内容的文本" if lang == "zh" else "Check if the text contains ID card related content and filter out texts that contain ID card related content."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='id_card_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []
        pattern = re.compile(r"(身\s{0,10}份|id\s{0,10}number\s{0,10}|identification|identity|\s{0,10}ID\s{0,10}No\s{0,10}|id\s{0,10}card\s{0,10}|NRIC\s{0,10}number\s{0,10}|IC\s{0,10}number\s{0,10}|resident\s{0,10}registration\s{0,10}|I.D.\s{0,10}Number\s{0,10})", re.I)

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                matches = pattern.findall(text)
                has_too_many_id_terms = len(matches) >= self.threshold
                valid_checks.append(not has_too_many_id_terms)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class NoPuncFilter(OperatorABC):

    def __init__(self, threshold: int=112):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否不含标点符号，过滤掉不含标点符号的文本" if lang == "zh" else "Check if the text does not contain punctuation marks and filter out texts that do not contain punctuation marks."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='no_punc_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                paragraphs = text.split('\n')
                max_word_count = 0
                for paragraph in paragraphs:
                    if len(paragraph.strip()) == 0:
                        continue
                    sentences = re.split("[–.!?,;•/|…]", paragraph)
                    for sentence in sentences:
                        words = sentence.split()
                        word_count = len(words)
                        if word_count > max_word_count:
                            max_word_count = word_count

                valid_checks.append(int(max_word_count) <= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class SpecialCharacterFilter(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否包含特殊字符，过滤掉包含特殊字符的文本" if lang == "zh" else "Check if the text contains special characters and filter out texts that contain special characters."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='special_character_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        speclai_character = [
            r"u200e",
            r"&#247;|\? :",
            r"[�□]|\{\/U\}",
            r"U\+26[0-F][0-D]|U\+273[3-4]|U\+1F[3-6][0-4][0-F]|U\+1F6[8-F][0-F]"
        ]

        valid_checks = []
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                # Check for special characters using regular expressions
                has_special_character = any(re.search(pattern, text) for pattern in speclai_character)
                valid_checks.append(not has_special_character)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class WatermarkFilter(OperatorABC):

    def __init__(self, watermarks: list= ['Copyright', 'Watermark', 'Confidential']):
        self.logger = get_logger()
        self.watermarks = watermarks
        self.logger.info(f"Initializing {self.__class__.__name__} with watermarks={self.watermarks}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否包含水印，过滤掉包含水印的文本" if lang == "zh" else "Check if the text contains watermarks and filter out texts that contain watermarks."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='watermark_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []
        
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                matches = re.search('|'.join(self.watermarks), text)
                valid_checks.append(matches is None)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class MeanWordLengthFilter(OperatorABC):

    def __init__(self, min_length: float=3, max_length: float=10):
        self.logger = get_logger()
        self.min_length = min_length
        self.max_length = max_length
        self.logger.info(f"Initializing {self.__class__.__name__} with min_length={self.min_length}, max_length={self.max_length}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的平均单词长度是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the average word length in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='mean_word_length_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []
        
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                normalized_words = text.split()
                num_words = len(normalized_words)

                if num_words == 0:
                    valid_checks.append(False)
                    continue

                num_chars = sum(len(word) for word in normalized_words)
                mean_length = round(num_chars / num_words, 2)

                valid_checks.append(self.min_length <= mean_length < self.max_length)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class StopWordFilter(OperatorABC):

    def __init__(self, threshold: float, use_tokenizer: bool):
        self.logger = get_logger()
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}, use_tokenizer = {self.use_tokenizer}...")
        import nltk
        # Download stopwords for the English language
        nltk.data.path.append('./dataflow/operators/filter/GeneralText/nltkdata/')
        nltk.download('stopwords', download_dir='./dataflow/operators/filter/GeneralText/nltkdata/')

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的停用词比例是否超过阈值，过滤掉不符合条件的文本" if lang == "zh" else "Check if the ratio of stop words in the text exceeds the threshold and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='stop_word_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                if self.use_tokenizer:
                    words = word_tokenize(text.lower())
                else:
                    words = text.lower().split()

                num_words = len(words)
                num_stop_words = sum(map(lambda w: w in self.stw, words))
                
                ratio = num_stop_words / num_words if num_words > 0 else 0

                valid_checks.append(ratio > self.threshold and num_stop_words > 2)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class CurlyBracketFilter(OperatorABC):

    def __init__(self, threshold: float=0.025):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold={self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中括号比例是否过高，过滤掉括号比例过高的文本" if lang == "zh" else "Check if the ratio of curly brackets in the text is too high and filter out texts with a high ratio of curly brackets."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='curly_bracket_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                num = text.count('{') + text.count('}')
                ratio = num / len(text) if len(text) != 0 else 0
                valid_checks.append(ratio < self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class CapitalWordsFilter(OperatorABC):

    def __init__(self, threshold: float=0.2, use_tokenizer: bool=False):
        self.logger = get_logger()
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}, use_tokenizer = {self.use_tokenizer}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的大写单词比例是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the ratio of capital words in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='capital_words_filter'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                if self.use_tokenizer:
                    words = word_tokenize(text)
                else:
                    words = text.split()

                num_words = len(words)
                num_caps_words = sum(map(str.isupper, words))

                ratio = num_caps_words / num_words if num_words > 0 else 0

                valid_checks.append(ratio <= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class LoremIpsumFilter(OperatorABC):

    def __init__(self, threshold: float=3e-8):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中是否包含Lorem Ipsum内容，过滤掉包含Lorem Ipsum内容的文本" if lang == "zh" else "Check if the text contains Lorem Ipsum content and filter out texts that contain Lorem Ipsum content."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='loremipsum_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        SEARCH_REGEX = re.compile(r"lorem ipsum", re.IGNORECASE)

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                normalized_content = text.lower()
                num_occurrences = len(SEARCH_REGEX.findall(normalized_content))

                ratio = num_occurrences / len(normalized_content) if len(normalized_content) > 0 else 0
                valid_checks.append(ratio <= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class UniqueWordsFilter(OperatorABC):

    def __init__(self, threshold: float=0.1):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的唯一单词比例是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the ratio of unique words in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='unique_words_filter'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                normalized_text = text.lower()
                normalized_words = tuple(normalized_text.split())
                num_normalized_words = len(normalized_words)

                if num_normalized_words == 0:
                    valid_checks.append(False)
                    continue

                num_unique_words = len(set(normalized_words))
                ratio = num_unique_words / num_normalized_words
                valid_checks.append(ratio > self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class CharNumberFilter(OperatorABC):

    def __init__(self, threshold: int=100):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold = {self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本中的字符数量是否在指定范围内，过滤掉不符合条件的文本" if lang == "zh" else "Check if the number of characters in the text is within a specified range and filter out texts that do not meet the criteria."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='char_number_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                # Remove whitespace and count the number of characters
                text = text.strip().replace(" ", "").replace("\n", "").replace("\t", "")
                num_char = len(text)

                # Check if the number of characters meets the threshold
                valid_checks.append(num_char >= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class LineStartWithBulletpointFilter(OperatorABC):

    def __init__(self, threshold: float=0.9):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold={self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本行是否以项目符号开头，过滤掉以项目符号开头的文本行" if lang == "zh" else "Check if the lines in the text start with bullet points and filter out lines that start with bullet points."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='line_start_with_bullet_point_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        valid_checks = []

        key_list = [
            "\u2022", "\u2023", "\u25B6", "\u25C0", "\u25E6", "\u25A0", "\u25A1", "\u25AA", "\u25AB", "\u2013"
        ]

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                raw_lines = split_paragraphs(text=text, normalizer=lambda x: x, remove_empty=True)
                num_lines = len(raw_lines)
                
                if num_lines == 0:
                    valid_checks.append(False)
                    continue

                num_occurrences = sum([line.text.lstrip().startswith(tuple(key_list)) for line in raw_lines])
                ratio = num_occurrences / num_lines
                valid_checks.append(ratio <= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class LineWithJavascriptFilter(OperatorABC):

    def __init__(self, threshold: int=3):
        self.logger = get_logger()
        self.threshold = threshold
        self.logger.info(f"Initializing {self.__class__.__name__} with threshold={self.threshold}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "检查文本行是否包含'javascript'，过滤掉包含'javascript'的文本行" if lang == "zh" else "Check if the lines in the text contain 'javascript' and filter out lines that contain 'javascript'."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='line_with_javascript_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        valid_checks = []

        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                normalized_lines = split_paragraphs(text=text, normalizer=normalize, remove_empty=True)
                num_lines = len(normalized_lines)

                if num_lines == 0:
                    valid_checks.append(False)
                    continue

                num_occurrences = sum(['javascript' in line.text.lower() for line in normalized_lines])
                num_not_occur = num_lines - num_occurrences

                valid_checks.append(num_lines <= 3 or num_not_occur >= self.threshold)
            else:
                valid_checks.append(False)

        valid_checks = np.array(valid_checks, dtype=int)

        # Filter the dataframe based on the result
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]

@OPERATOR_REGISTRY.register()
class BlocklistFilter(OperatorABC):

    def __init__(self, language:str = 'en', threshold:int = 1, use_tokenizer:bool = False):
        self.logger = get_logger()
        self.language = language
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.logger.info(f"Initializing {self.__class__.__name__} with language = {self.language}, threshold = {self.threshold}, use_tokenizer = {self.use_tokenizer}...")
        self.blocklist = self.load_blocklist()

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用预定义的阻止词列表过滤文本" if lang == "zh" else "Filter text using a predefined blocklist of words."

    def load_blocklist(self):
        dataflow_dir = DataFlowPath.get_dataflow_dir()
        file_path = f"{dataflow_dir}/operators/filter/GeneralText/blocklist/{self.language}.txt"
        self.logger.info(f"Loading blocklist for language '{self.language}' from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            blocklist = set(line.strip().lower() for line in file if line.strip())
        self.logger.info(f"Blocklist for '{self.language}' loaded. Total words in blocklist: {len(blocklist)}.")
        return blocklist

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'blocklist_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")
        valid_checks = []
        for text in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            if text:
                if self.use_tokenizer:
                    text = word_tokenize(text.lower())
                else:
                    text = text.lower().split()
                blocklist_count = sum(1 for word in text if word in self.blocklist)
                valid_checks.append(blocklist_count <= self.threshold)
        valid_checks = np.array(valid_checks, dtype=int)
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
