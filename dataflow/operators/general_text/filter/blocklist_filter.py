from tqdm import tqdm
import numpy as np
import nltk
import os
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.cli_funcs.paths import DataFlowPath
from nltk.tokenize import word_tokenize

@OPERATOR_REGISTRY.register()
class BlocklistFilter(OperatorABC):

    def __init__(self, language:str = 'en', threshold:int = 1, use_tokenizer:bool = False):
        self.logger = get_logger()
        self.language = language
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.logger.info(f"Initializing {self.__class__.__name__} with language = {self.language}, threshold = {self.threshold}, use_tokenizer = {self.use_tokenizer}...")
        
        # 设置 NLTK 数据路径（如果环境变量中有的话）
        if 'NLTK_DATA' in os.environ:
            nltk.data.path.insert(0, os.environ['NLTK_DATA'])
        
        # 尝试查找数据，如果不存在则下载
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.blocklist = self.load_blocklist()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子使用特定语言的阻止列表进行文本过滤，支持可选的分词器进行单词级匹配。\n"
                "输入参数：\n"
                "- input_key：输入文本字段名（必填）\n"
                "- language：语言代码，默认为'en'\n"
                "- threshold：匹配次数阈值，默认为1\n"
                "- use_tokenizer：是否使用分词器，默认为False\n"
                "- output_key：写入过滤标签的输出字段名，默认为'blocklist_filter_label'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留阻止列表命中次数<=threshold的非空文本行；"
                "每行新增output_key字段，值为整数标签（1=通过，0=未通过；由于只保留通过的行，输出中该值均为1），该字段保留在输出中\n"
                "- 返回包含output_key的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator filters text using language-specific blocklists with optional tokenizer integration for word-level filtering.\n"
                "Input Parameters:\n"
                "- input_key: Input text field name (required)\n"
                "- language: Language code, default is 'en'\n"
                "- threshold: Matching count threshold, default is 1\n"
                "- use_tokenizer: Whether to use tokenizer, default is False\n"
                "- output_key: Name of the output field that receives the filter label, default is 'blocklist_filter_label'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only non-empty rows whose blocklist hit count is <= threshold; "
                "each row gets a new output_key field holding an integer label (1 = pass, 0 = fail; since only passing rows are kept, it is 1 in the output), which is kept in the output\n"
                "- List containing output_key for subsequent operator reference"
            )
        else:
            return "BlocklistFilter uses language-specific blocklists with optional tokenizer integration."
        
    def load_blocklist(self):
        dataflow_dir = DataFlowPath.get_dataflow_dir()
        file_path = f"{dataflow_dir}/operators/general_text/filter/blocklist/{self.language}.txt"
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
            else:
                valid_checks.append(0)
        valid_checks = np.array(valid_checks, dtype=int)
        dataframe[self.output_key] = valid_checks
        filtered_dataframe = dataframe[valid_checks == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]