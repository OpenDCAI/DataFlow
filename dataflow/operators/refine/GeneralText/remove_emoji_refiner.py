import re
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register()
class RemoveEmojiRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.refiner_name = 'RemoveEmojiRefiner'
        self.logger.info(f"Initializing {self.__class__.__name__} ...")

        # Emoji pattern for matching emojis in the text
        self.emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Miscellaneous symbols and pictographs
            u"\U0001F680-\U0001F6FF"  # Transport and map symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", 
            flags=re.UNICODE
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "去除文本中的图像表情符号" if lang == "zh" else "Remove emojis from the text."

    def run(self, storage: DataFlowStorage, input_key: str):
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {input_key}...")

        for item in tqdm(dataframe[input_key], desc=f"Implementing {self.refiner_name}"):
            modified = False
            original_text = item
            no_emoji_text = self.emoji_pattern.sub(r'', original_text)

            if original_text != no_emoji_text:
                item = no_emoji_text
                modified = True
                self.logger.debug(f"Modified text for key '{input_key}': Original: {original_text[:30]}... -> Refined: {no_emoji_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataframe[input_key] = refined_data
        storage.write(dataframe)
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")

        return [input_key]
