import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class HtmlUrlRemoverRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "去除文本中的URL链接和HTML标签，净化文本内容。使用正则表达式匹配并移除各种形式的URL和HTML标签。"
                "输入参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含净化后文本的DataFrame\n"
                "- 返回输入字段名，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Remove URL links and HTML tags from text to clean content. Uses regular expressions to match and remove various forms of URLs and HTML tags.\n"
                "Input Parameters:\n"
                "- input_key: Field name for input text\n\n"
                "Output Parameters:\n"
                "- DataFrame containing cleaned text\n"
                "- Returns input field name for subsequent operator reference"
            )
        else:
            return (
                "HtmlUrlRemoverRefiner cleans text by removing URLs and HTML tags."
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
            refined_text = original_text

            # Remove URLs
            refined_text = re.sub(r'https?:\/\/\S+[\r\n]*', '', refined_text, flags=re.MULTILINE)
            # Remove HTML tags
            refined_text = re.sub(r'<.*?>', '', refined_text)

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