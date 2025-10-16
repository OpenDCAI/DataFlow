from dataflow.core import OperatorABC
from typing import Callable, Tuple
import numpy as np
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import re

@OPERATOR_REGISTRY.register()
class ColonEndFilter(OperatorABC):

    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于检查文本是否以冒号结尾，常用于判断问题是否为不完整的提问。\n"
                "初始化参数：\n"
                "- 无\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'{类名小写}_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if text ends with a colon, commonly used to identify incomplete questions.\n"
                "Initialization Parameters:\n"
                "- None\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is '{classname_lower}_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "ColonEndFilter checks if text ends with a colon and filters out incomplete questions."
        
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
class SentenceNumberFilter(OperatorABC):

    def __init__(self, min_sentences: int=3, max_sentences: int=7500):
        self.logger = get_logger()
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.logger.info(f"Initializing {self.__class__.__name__} with min_sentences = {self.min_sentences}, max_sentences = {self.max_sentences}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于检查文本中的句子数量是否在指定范围内，使用正则表达式匹配句子结束符号(。！？.!?)进行分割。\n"
                "初始化参数：\n"
                "- min_sentences：最小句子数量阈值，默认为3\n"
                "- max_sentences：最大句子数量阈值，默认为7500\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'sentence_number_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if the number of sentences in text is within specified range, using regex to match sentence-ending punctuation(。！？.!?).\n"
                "Initialization Parameters:\n"
                "- min_sentences: Minimum sentence count threshold, default is 3\n"
                "- max_sentences: Maximum sentence count threshold, default is 7500\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'sentence_number_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "SentenceNumberFilter filters text based on sentence count range using regex pattern matching."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并过滤以省略号(...)或(……)结尾的文本行，常用于识别不完整的表述。\n"
                "初始化参数：\n"
                "- threshold：以省略号结尾的行数比率阈值，默认为0.3\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'line_end_with_ellipsis_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and filters text lines ending with ellipsis (...) or (……), commonly used to identify incomplete statements.\n"
                "Initialization Parameters:\n"
                "- threshold: Ratio threshold of lines ending with ellipsis, default is 0.3\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'line_end_with_ellipsis_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "LineEndWithEllipsisFilter detects and filters text ending with ellipsis characters."
        
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
        if lang == "zh":
            return (
                "该算子用于过滤空值、空字符串或仅包含空白字符的文本，确保输入数据的有效性。\n"
                "初始化参数：\n"
                "- 无\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'content_null_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator filters null values, empty strings, or text containing only whitespace characters to ensure data validity.\n"
                "Initialization Parameters:\n"
                "- None\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'content_null_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "ContentNullFilter removes null, empty, and whitespace-only text content."
        
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
        if lang == "zh":
            return (
                "该算子用于检查文本中特定符号(#, ..., …)与单词数量的比率是否超过阈值，过滤符号使用过多的文本。\n"
                "初始化参数：\n"
                "- threshold：符号与单词比率阈值，默认为0.4\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'symbol_word_ratio_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if the ratio of specific symbols(#, ..., …) to word count exceeds threshold, filtering text with excessive symbol usage.\n"
                "Initialization Parameters:\n"
                "- threshold: Symbol-to-word ratio threshold, default is 0.4\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'symbol_word_ratio_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "SymbolWordRatioFilter checks ratio of specified symbols to word count and filters excessive usage."
        
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
        if lang == "zh":
            return (
                "该算子用于验证文本中字母单词的比率是否达到阈值，支持NLTK分词或简单空格分割两种模式。\n"
                "初始化参数：\n"
                "- threshold：字母单词比率阈值（无默认值，必须提供）\n"
                "- use_tokenizer：是否使用NLTK分词器（无默认值，必须提供）\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'alpha_words_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator verifies if the ratio of alphabetic words in text meets threshold, supporting NLTK tokenization or simple space splitting.\n"
                "Initialization Parameters:\n"
                "- threshold: Alphabetic word ratio threshold (no default, required)\n"
                "- use_tokenizer: Whether to use NLTK tokenizer (no default, required)\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'alpha_words_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "AlphaWordsFilter verifies alphabetic word ratio using either NLTK tokenization or space splitting."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并过滤包含HTML实体（如&amp;、&lt;、&gt;等）的文本，确保内容不包含标记语言元素。\n"
                "初始化参数：\n"
                "- 无\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'html_entity_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and filters text containing HTML entities (e.g., &amp;, &lt;, &gt;) to ensure content has no markup language elements.\n"
                "Initialization Parameters:\n"
                "- None\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'html_entity_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "HtmlEntityFilter detects and removes text containing HTML entity patterns."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并过滤包含身份证相关术语的文本，使用正则表达式匹配身份证号码模式以保护敏感信息。\n"
                "初始化参数：\n"
                "- threshold：身份证相关词汇匹配次数阈值，默认为3\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'id_card_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and filters text containing ID card-related terms using regex patterns to protect sensitive information.\n"
                "Initialization Parameters:\n"
                "- threshold: ID card-related terms matching count threshold, default is 3\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'id_card_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "IDCardFilter detects and removes text containing ID card numbers and related sensitive information."
        
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
        if lang == "zh":
            return (
                "该算子用于确保文本包含足够的标点符号，通过统计句子间最大单词数量进行过滤。\n"
                "初始化参数：\n"
                "- threshold：句子间最大单词数量阈值，默认为112\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'no_punc_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator ensures text contains sufficient punctuation by counting maximum word count between sentences.\n"
                "Initialization Parameters:\n"
                "- threshold: Maximum word count between sentences threshold, default is 112\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'no_punc_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "NoPuncFilter ensures text contains sufficient punctuation marks based on ratio threshold."
        
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
        if lang == "zh":
            return (
                "该算子用于移除包含特殊/unicode字符的文本，使用预定义模式检测非标准字符以确保文本规范性。\n"
                "初始化参数：\n"
                "- 无\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'special_character_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator removes text containing special/unicode characters using predefined patterns to ensure text normalization.\n"
                "Initialization Parameters:\n"
                "- None\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'special_character_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "SpecialCharacterFilter removes text containing special or non-standard unicode characters."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并移除包含版权/水印内容的文本，使用指定关键词列表识别受保护内容。\n"
                "初始化参数：\n"
                "- watermarks：水印关键词列表，默认为['Copyright', 'Watermark', 'Confidential']\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'watermark_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and removes copyrighted/watermarked content using specified keyword lists to identify protected material.\n"
                "Initialization Parameters:\n"
                "- watermarks: List of watermark keywords, default is ['Copyright', 'Watermark', 'Confidential']\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'watermark_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "WatermarkFilter detects and removes text containing copyright or watermark keywords."
        
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
        if lang == "zh":
            return (
                "该算子用于检查文本中单词的平均长度是否在指定范围内，通过字符总数除以单词数量计算平均值。\n"
                "初始化参数：\n"
                "- min_length：最小平均单词长度，默认为3\n"
                "- max_length：最大平均单词长度，默认为10\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'mean_word_length_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if the average word length in text is within specified range, calculated by total characters divided by word count.\n"
                "Initialization Parameters:\n"
                "- min_length: Minimum average word length, default is 3\n"
                "- max_length: Maximum average word length, default is 10\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'mean_word_length_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "MeanWordLengthFilter checks average word length against specified range using character and word counts."
        
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
        if lang == "zh":
            return (
                "该算子用于验证文本中停用词的比率是否高于阈值，使用NLTK分词器进行单词分割和停用词识别。\n"
                "初始化参数：\n"
                "- threshold：停用词比率阈值（无默认值，必须提供）\n"
                "- use_tokenizer：是否使用NLTK分词器（无默认值，必须提供）\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'stop_word_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator verifies if the ratio of stop words in text is above threshold, using NLTK tokenizer for word splitting and stop word identification.\n"
                "Initialization Parameters:\n"
                "- threshold: Stop word ratio threshold (no default, required)\n"
                "- use_tokenizer: Whether to use NLTK tokenizer (no default, required)\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'stop_word_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "StopWordFilter verifies stop word ratio using NLTK tokenization with configurable threshold."
        
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
        if lang == "zh":
            return (
                "该算子用于检测文本中是否存在过多的花括号使用，通过花括号数量与文本长度的比率进行过滤。\n"
                "初始化参数：\n"
                "- threshold：花括号比率阈值，默认为0.025\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'curly_bracket_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects excessive curly bracket usage in text by comparing bracket count to text length ratio.\n"
                "Initialization Parameters:\n"
                "- threshold: Bracket ratio threshold, default is 0.025\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'curly_bracket_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "CurlyBracketFilter detects excessive curly bracket usage with ratio thresholding."
        
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
        if lang == "zh":
            return (
                "该算子用于检查文本中大写单词的比率是否超过阈值，支持可选的分词器进行单词识别。\n"
                "初始化参数：\n"
                "- threshold：大写单词比率阈值，默认为0.2\n"
                "- use_tokenizer：是否使用NLTK分词器，默认为False\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'capital_words_filter'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if the ratio of capitalized words in text exceeds threshold, supporting optional tokenizer for word identification.\n"
                "Initialization Parameters:\n"
                "- threshold: Capitalized word ratio threshold, default is 0.2\n"
                "- use_tokenizer: Whether to use NLTK tokenizer, default is False\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'capital_words_filter'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "CapitalWordsFilter checks uppercase word ratio with optional tokenizer usage."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并过滤包含占位文本（如'lorem ipsum'）的文本，使用正则表达式模式匹配并结合阈值过滤。\n"
                "初始化参数：\n"
                "- threshold：'lorem ipsum'出现次数与文本长度的比率阈值，默认为3e-8\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'loremipsum_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and filters text containing placeholder text (e.g., 'lorem ipsum') using regex pattern matching with threshold filtering.\n"
                "Initialization Parameters:\n"
                "- threshold: Ratio threshold of 'lorem ipsum' occurrences to text length, default is 3e-8\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'loremipsum_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "LoremIpsumFilter detects and removes text containing placeholder text patterns."
        
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
        if lang == "zh":
            return (
                "该算子用于检查文本中唯一单词的比率是否达到阈值，通过集合操作计算唯一单词数量与总单词数量的比率。\n"
                "初始化参数：\n"
                "- threshold：最小唯一单词比率阈值，默认为0.1\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'unique_words_filter'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator checks if the ratio of unique words in text meets threshold, calculating ratio of unique word count to total word count using set operations.\n"
                "Initialization Parameters:\n"
                "- threshold: Minimum unique word ratio threshold, default is 0.1\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'unique_words_filter'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "UniqueWordsFilter checks unique word ratio using set operations and threshold comparison."
        
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
        if lang == "zh":
            return (
                "该算子用于验证文本在去除空白字符后的字符数量是否达到最小阈值。\n"
                "初始化参数：\n"
                "- threshold：最小字符数量阈值，默认为100\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'char_number_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator verifies if the character count of text (after whitespace removal) meets minimum threshold.\n"
                "Initialization Parameters:\n"
                "- threshold: Minimum character count threshold, default is 100\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'char_number_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "CharNumberFilter verifies character count after whitespace removal against specified threshold."
        
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
        if lang == "zh":
            return (
                "该算子用于检测并过滤以各种项目符号符号开头的文本行，使用Unicode字符匹配结合比率阈值进行过滤。\n"
                "初始化参数：\n"
                "- threshold：以项目符号开头的行数比率阈值，默认为0.9\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'line_start_with_bullet_point_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator detects and filters lines starting with various bullet point symbols using Unicode character matching with ratio thresholding.\n"
                "Initialization Parameters:\n"
                "- threshold: Ratio threshold of lines starting with bullet points, default is 0.9\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'line_start_with_bullet_point_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "LineStartWithBulletpointFilter detects various bullet point symbols using Unicode character matching."
        
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
        if lang == "zh":
            return (
                "该算子用于识别并过滤包含'javascript'引用的文本，通过关键词匹配和阈值判断进行内容过滤。\n"
                "初始化参数：\n"
                "- threshold：不包含'javascript'的最小行数阈值，默认为3\n"
                "运行参数：\n"
                "- storage：DataFlowStorage对象\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出标签字段名，默认为'line_with_javascript_filter_label'\n"
                "返回值：\n"
                "- 包含output_key的列表"
            )
        elif lang == "en":
            return (
                "This operator identifies and filters text containing 'javascript' references through keyword matching and threshold judgment.\n"
                "Initialization Parameters:\n"
                "- threshold: Minimum line count threshold without 'javascript', default is 3\n"
                "Run Parameters:\n"
                "- storage: DataFlowStorage object\n"
                "- input_key: Input text field name\n"
                "- output_key: Output label field name, default is 'line_with_javascript_filter_label'\n"
                "Returns:\n"
                "- List containing output_key"
            )
        else:
            return "LineWithJavascriptFilter identifies 'javascript' references in text with threshold-based filtering."
        
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
