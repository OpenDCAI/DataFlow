import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class LowercaseRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "将文本字段中的所有大写字符转换为小写，统一文本格式。对指定字段的文本内容进行全小写处理。"
                "输入参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含小写转换后文本的DataFrame\n"
                "- 返回输入字段名，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Convert all uppercase characters in text fields to lowercase to unify text format. Applies full lowercase processing to text content of specified fields.\n"
                "Input Parameters:\n"
                "- input_key: Field name for input text\n\n"
                "Output Parameters:\n"
                "- DataFrame containing lowercase converted text\n"
                "- Returns input field name for subsequent operator reference"
            )
        else:
            return (
                "LowercaseRefiner converts text fields to lowercase."
            )
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False  
            original_text = item
            lower_text = original_text.lower()
            if original_text != lower_text:
                item = lower_text
                modified = True  
            self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {lower_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]