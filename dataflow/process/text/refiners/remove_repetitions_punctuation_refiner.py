from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemoveRepetitionsPunctuationRefiner class is a text refiner that removes repeated punctuation characters 
from specified text fields in a dataset. Using a regular expression, it reduces consecutive occurrences of the 
same punctuation mark (e.g., "!!" becomes "!") to a single instance, including repeated underscores.

This refiner is useful for cleaning up text that may have excessive or stylistic punctuation, which can interfere 
with analysis or make the text harder to read. After processing, the refiner returns the modified dataset along 
with a count of the items that were altered, resulting in a more uniform text format with minimal punctuation repetition.
"""

@PROCESSOR_REGISTRY.register()
class RemoveRepetitionsPunctuationRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveRepetitionsPunctuationRefiner'

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
                        no_extra_punct_text = re.sub(r'([^\w\s_])\1+|(_)\2+', r'\1\2', original_text)
                             
                        if original_text != no_extra_punct_text:
                            item[key] = no_extra_punct_text
                            modified = True  

                refined_data.append(item)
                if modified:
                    numbers += 1
        dataset.dataset = refined_data
        return dataset, numbers