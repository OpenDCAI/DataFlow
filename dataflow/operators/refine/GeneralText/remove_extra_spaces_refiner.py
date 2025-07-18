import re
from tqdm import tqdm
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveExtraSpacesRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于移除文本中的多余空格，将连续的多个空格替换为单个空格，并去除文本前后的空白字符。\n"
                "通过字符串分割和连接实现空格标准化，提高文本格式一致性。\n"
                "输入参数：\n"
                "- 无初始化参数\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含标准化空格的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator removes extra spaces from text, replacing consecutive spaces with single spaces and trimming leading/trailing whitespace.\n"
                "Achieves space standardization through string splitting and joining to improve text format consistency.\n"
                "Input Parameters:\n"
                "- No initialization parameters\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with standardized spacing\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Removes extra spaces and normalizes spacing in text."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        numbers = 0
        refined_data = []

        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            refined_text = " ".join(original_text.split())  # Remove extra spaces

            if original_text != refined_text:
                item = refined_text
                modified = True
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {refined_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataframe[self.input_key] = refined_data
        storage.write(dataframe)
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")

        return [self.input_key]
