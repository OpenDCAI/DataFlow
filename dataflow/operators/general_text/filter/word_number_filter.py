from tqdm import tqdm
import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class WordNumberFilter(OperatorABC):

    def __init__(self, min_words: int=20, max_words: int=100000):
        self.logger = get_logger()
        self.min_words = min_words
        self.max_words = max_words
        self.logger.info(f"Initializing {self.__class__.__name__} with min_words = {self.min_words}, max_words = {self.max_words}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于过滤单词数量不在指定范围内的文本，通过空格分割计算单词数量。\n"
                "输入参数：\n"
                "- input_key：输入文本字段名，默认为'text'\n"
                "- min_words：最小单词数量阈值，默认为5\n"
                "- max_words：最大单词数量阈值，默认为100\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留单词数量在指定范围内的文本行\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator filters text with word count outside the specified range, using space splitting for word counting.\n"
                "Input Parameters:\n"
                "- input_key: Input text field name, default is 'text'\n"
                "- min_words: Minimum word count threshold, default is 5\n"
                "- max_words: Maximum word count threshold, default is 100\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only rows with word count within specified range\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "WordNumberFilter filters text based on word count range using space splitting."
        
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

