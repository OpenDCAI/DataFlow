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
        return "规范化文本中的日期格式、货币格式等" if lang == "zh" else "Normalize date, currency, and other formats in text."
    
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