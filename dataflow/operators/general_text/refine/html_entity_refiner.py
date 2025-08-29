import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class HtmlEntityRefiner(OperatorABC):
    def __init__(self, html_entities: list = [
            "nbsp", "lt", "gt", "amp", "quot", "apos", "hellip", "ndash", "mdash", 
            "lsquo", "rsquo", "ldquo", "rdquo"
        ]):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        # 从参数中获取自定义 HTML 实体列表，如果未提供则使用默认列表
        self.html_entities = html_entities

        # 构建正则表达式模式，匹配所有定义的 HTML 实体
        # 包括以下几种形式：
        # 1. &实体名;
        # 2. ＆实体名; （全角 &）
        # 3. &实体名； （中文分号）
        # 4. ＆实体名； （全角 & + 中文分号）
        entity_patterns = []
        for entity in self.html_entities:
            # &实体名;
            entity_patterns.append(fr'&{entity};')
            # ＆实体名; （全角 &）
            entity_patterns.append(fr'＆{entity};')
            # &实体名； （中文分号）
            entity_patterns.append(fr'&{entity}；')
            # ＆实体名； （全角 & + 中文分号）
            entity_patterns.append(fr'＆{entity}；')

        # 编译正则表达式
        self.html_entity_regex = re.compile('|'.join(entity_patterns))

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "去除文本中的HTML实体，包括标准实体（如&nbsp;、&lt;）和各种变体形式（全角符号、中文分号等）。支持自定义需要移除的HTML实体列表。"
                "输入参数：\n"
                "- html_entities：需要移除的HTML实体列表，默认为包含常见实体的列表\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含移除HTML实体后文本的DataFrame\n"
                "- 返回输入字段名，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Remove HTML entities from text, including standard entities (e.g., &nbsp;, &lt;) and various variants (full-width symbols, Chinese semicolons, etc.). \n"
                "Supports custom list of HTML entities to be removed.\n"
                "Input Parameters:\n"
                "- html_entities: List of HTML entities to remove, default contains common entities\n"
                "- input_key: Field name for input text\n\n"
                "Output Parameters:\n"
                "- DataFrame containing text with HTML entities removed\n"
                "- Returns input field name for subsequent operator reference"
            )
        else:
            return (
                "HtmlEntityRefiner removes HTML entities and their variants from text."
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

            # 使用正则表达式替换所有匹配的HTML实体为空字符串
            refined_text = self.html_entity_regex.sub('', refined_text)

            # 检查文本是否被修改
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