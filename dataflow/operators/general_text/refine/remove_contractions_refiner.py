import contractions
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveContractionsRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于扩展文本中的英语缩写词，将缩写形式转换为完整形式（例如将\"can't\"扩展为\"cannot\"）。\n"
                "使用contractions库进行缩写词扩展，提高文本标准化程度。\n"
                "输入参数：\n"
                "- 无初始化参数\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含扩展缩写词后的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator expands English contractions in text, converting abbreviated forms to full forms (e.g., \"can't\" → \"cannot\").\n"
                "Uses the contractions library for abbreviation expansion to improve text standardization.\n"
                "Input Parameters:\n"
                "- No initialization parameters\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with expanded contractions\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Expands English contractions in text to improve standardization."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            expanded_text = contractions.fix(original_text)
            if original_text != expanded_text:
                item = expanded_text
                modified = True
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {expanded_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]