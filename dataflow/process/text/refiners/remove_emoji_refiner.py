from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm
from dataflow.utils.utils import get_logger


class RemoveEmojiRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.refiner_name = 'RemoveEmojiRefiner'
        self.logger.info(f"Initializing {self.refiner_name}...")
        
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

    def refine_func(self, dataset):
        self.logger.info(f"Start running {self.refiner_name}...")
        refined_data = []
        numbers = 0
        keys = dataset.keys if isinstance(dataset.keys, list) else [dataset.keys]

        for item in tqdm(dataset, desc=f"Implementing {self.refiner_name}"):
            if isinstance(item, dict):
                modified = False  
                for key in keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        no_emoji_text = self.emoji_pattern.sub(r'', original_text)

                        if original_text != no_emoji_text:
                            item[key] = no_emoji_text
                            modified = True  
                            self.logger.debug(f"Modified text for key '{key}': Original: {original_text[:30]}... -> Refined: {no_emoji_text[:30]}...")

                refined_data.append(item)
                if modified:
                    numbers += 1
                    self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataset.dataset = refined_data
        self.logger.info(f"Refining completed. Total items modified: {numbers}")
        return dataset, numbers
