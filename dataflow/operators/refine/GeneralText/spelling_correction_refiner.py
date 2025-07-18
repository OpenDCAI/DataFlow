import re
import os
import requests
from tqdm import tqdm
from symspellpy.symspellpy import SymSpell, Verbosity
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class SpellingCorrectionRefiner(OperatorABC):
    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7, dictionary_path: str = "frequency_dictionary_en_82_765.txt"):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.max_edit_distance = max_edit_distance  # Default to 2 if not specified
        self.prefix_length = prefix_length  # Default to 7 if not specified
        self.dictionary_path = dictionary_path
        # If dictionary is not found locally, download it
        if not os.path.exists(self.dictionary_path):
            self.download_dictionary()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=self.prefix_length)
        term_index = 0
        count_index = 1
        if not self.sym_spell.load_dictionary(self.dictionary_path, term_index, count_index):
            self.logger.error(f"Error loading dictionary at {self.dictionary_path}")
        self.logger.info(f"Successfully loaded dictionary at {self.dictionary_path}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于通过SymSpell算法对文本中的拼写错误进行纠正，支持自定义编辑距离和词典路径。\n"
                "若本地词典不存在则自动下载，使用近似字符串匹配实现拼写纠错功能。\n"
                "输入参数：\n"
                "- max_edit_distance：最大编辑距离，默认为2\n"
                "- prefix_length：前缀长度，默认为7\n"
                "- dictionary_path：词典路径，默认为'frequency_dictionary_en_82_765.txt'\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含纠正拼写错误的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator corrects spelling errors in text using the SymSpell algorithm, supporting custom edit distance and dictionary path.\n"
                "Automatically downloads dictionary if not locally available, using approximate string matching for spelling correction.\n"
                "Input Parameters:\n"
                "- max_edit_distance: Maximum edit distance, default is 2\n"
                "- prefix_length: Prefix length, default is 7\n"
                "- dictionary_path: Dictionary path, default is 'frequency_dictionary_en_82_765.txt'\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with corrected spelling\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Corrects spelling errors in text using the SymSpell algorithm with configurable parameters."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            refined_text = self.spelling_checks(original_text)
            if original_text != refined_text:
                item = refined_text
                modified = True
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {refined_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]
    
    def spelling_checks(self, text):
        correct_result = []
        for word in text.split():
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, self.max_edit_distance)
            corrected_word = suggestions[0].term if suggestions else word
            correct_result.append(corrected_word)

        return " ".join(correct_result)

    def download_dictionary(self):
        url = 'https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt'
        
        try:
            print("Downloading dictionary...")
            response = requests.get(url)
            response.raise_for_status() 
            
            with open(self.dictionary_path, 'wb') as file:
                file.write(response.content)
            print(f"Dictionary downloaded and saved to {self.dictionary_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dictionary: {e}")
    