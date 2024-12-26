from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
This refiner class, HtmlUrlRemoverRefiner, is designed to clean text data by removing URLs and HTML tags. 
It iterates over specified fields in a dataset, detects and removes any web URLs (e.g., starting with "http" or "https") 
and HTML elements (e.g., "<tag>"). After cleaning, it returns the refined dataset and counts how many items were modified.
"""

@PROCESSOR_REGISTRY.register()
class HtmlUrlRemoverRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'HtmlUrlRemoverRefiner'

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
                        refined_text = original_text

                        refined_text = re.sub(r'https?:\/\/\S+[\r\n]*', '', refined_text, flags=re.MULTILINE)
                        refined_text = re.sub(r'<.*?>', '', refined_text)
                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers
