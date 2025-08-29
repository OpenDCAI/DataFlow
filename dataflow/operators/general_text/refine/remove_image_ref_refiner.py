import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveImageRefsRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.image_pattern = re.compile(
            r'!\[\]\(images\/[0-9a-fA-F]\.jpg\)|'
            r'[a-fA-F0-9]+\.[a-zA-Z]{3,4}\)|'
            r'!\[\]\(images\/[a-f0-9]|'
            r'图\s+\d+-\d+：[\u4e00-\u9fa5a-zA-Z0-9]+|'
            r'(?:[0-9a-zA-Z]+){7,}|'                # 正则5
            r'(?:[一二三四五六七八九十零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+){5,}|'  # 正则6（汉字数字）
            r"u200e|"
            r"&#247;|\? :|"
            r"[�□]|\{\/U\}|"
            r"U\+26[0-F][0-D]|U\+273[3-4]|U\+1F[3-6][0-4][0-F]|U\+1F6[8-F][0-F]"
        )
        self.logger.info(f"Initializing {self.__class__.__name__} ...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于去除文本中的图片引用格式，包括Markdown图片链接、图片编号、特殊符号组合等图像引用模式。\n"
                "通过多模式正则表达式匹配，识别并移除多种图片引用格式。\n"
                "输入参数：\n"
                "- 无初始化参数\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含去除图片引用的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator removes image reference formats from text, including Markdown image links, image numbers, special symbol combinations and other image reference patterns.\n"
                "Identifies and removes multiple image reference formats through multi-pattern regular expression matching.\n"
                "Input Parameters:\n"
                "- No initialization parameters\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with image references removed\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Removes image reference formats from text using regular expressions."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            # 移除所有图片引用格式[1,2](@ref)
            cleaned_text = self.image_pattern.sub('', original_text)
            
            if original_text != cleaned_text:
                item = cleaned_text
                modified = True
                # 调试日志：显示修改前后的对比
                self.logger.debug(f"Modified text for key '{self.input_key}':")
                self.logger.debug(f"Original: {original_text[:100]}...")
                self.logger.debug(f"Refined : {cleaned_text[:100]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]