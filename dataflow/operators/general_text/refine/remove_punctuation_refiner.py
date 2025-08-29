import string
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemovePunctuationRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.punct_to_remove = string.punctuation
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于移除文本中的标点符号，包括英文标点符号集合中的所有符号。\n"
                "使用string.punctuation定义的标点集合进行过滤，实现文本去标点处理。\n"
                "输入参数：\n"
                "- 无初始化参数\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含去除标点的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator removes punctuation from text, including all symbols in the English punctuation set.\n"
                "Uses the punctuation set defined by string.punctuation for filtering to achieve text punctuation removal.\n"
                "Input Parameters:\n"
                "- No initialization parameters\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with punctuation removed\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Removes punctuation from text using string.punctuation set."
        
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            no_punct_text = original_text.translate(str.maketrans('', '', self.punct_to_remove))
            
            if original_text != no_punct_text:
                item = no_punct_text
                modified = True  
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {no_punct_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]