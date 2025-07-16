import re
import string
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveRepetitionsPunctuationRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.punct_to_remove = string.punctuation
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "移除重复的标点符号，例如“!!!”变为“!”" if lang == "zh" else "Remove repeated punctuation (e.g., '!!!' to '!')."
        
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            no_extra_punct_text = re.sub(r'([^\w\s_])\1+|(_)\2+', r'\1\2', original_text)
                    
            if original_text != no_extra_punct_text:
                item = no_extra_punct_text
                modified = True  

                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {no_extra_punct_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]