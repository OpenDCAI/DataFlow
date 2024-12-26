from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemoveEmojiRefiner class is a text refiner that removes emojis from specified text fields within a dataset. 
Using a predefined regex pattern, it identifies and removes common emoji characters across a range of Unicode blocks, 
including emoticons, flags, symbols, and various other pictographs. 

This class is useful for datasets where emojis may interfere with text analysis or processing. 
After removing emojis, the modified dataset is returned along with a count of items that were changed, 
ensuring cleaner and more standardized text content.
"""

@PROCESSOR_REGISTRY.register()
class RemoveEmojiRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveEmojiRefiner'
        self.emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )

    def refine_func(self, dataset):
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

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers
