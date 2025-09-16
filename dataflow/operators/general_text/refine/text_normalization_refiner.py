import re
from datetime import datetime
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class TextNormalizationRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于规范化文本中的日期格式和货币格式，统一为标准表示形式。\n"
                "日期格式统一转换为'YYYY-MM-DD'形式，货币格式转换为'金额 USD'形式，提高数据一致性。\n"
                "输入参数：\n"
                "- 无初始化参数\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含格式规范化的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator normalizes date formats and currency formats in text to standard representations.\n"
                "Unifies date formats to 'YYYY-MM-DD' and currency formats to 'amount USD' to improve data consistency.\n"
                "Input Parameters:\n"
                "- No initialization parameters\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with normalized formats\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Normalizes date and currency formats in text to standard representations."
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            refined_text = original_text

            refined_text = re.sub(r'(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})', r'\3-\2-\1', refined_text)
            date_patterns = [
                (r'\b(\w+)\s+(\d{1,2}),\s+(\d{4})\b', '%B %d, %Y'),
                (r'\b(\d{1,2})\s+(\w+)\s+(\d{4})\b', '%d %B %Y')
            ]
            for pattern, date_format in date_patterns:
                match = re.search(pattern, refined_text)
                if match:
                    date_str = match.group(0)
                    try:
                        parsed_date = datetime.strptime(date_str, date_format)
                        refined_text = refined_text.replace(date_str, parsed_date.strftime('%Y-%m-%d'))
                    except ValueError:
                        pass

            refined_text = re.sub(r'\$\s?(\d+)', r'\1 USD', refined_text)

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