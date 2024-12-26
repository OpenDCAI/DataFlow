from dataflow.core import TextRefiner
import string
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemovePunctuationRefiner class is a text refiner that removes all punctuation characters from specified text fields in a dataset.
Using Python’s `string.punctuation`, it identifies and removes common punctuation marks such as periods, commas, question marks, 
and exclamation points. This is useful in cases where punctuation might interfere with text analysis or processing.

After removing punctuation, the refiner returns the modified dataset along with a count of the modified items, resulting in 
a cleaner text format without any punctuation marks.
"""

@PROCESSOR_REGISTRY.register()
class RemovePunctuationRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemovePunctuationRefiner'
        self.punct_to_remove = string.punctuation

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
                        no_punct_text = original_text.translate(str.maketrans('', '', self.punct_to_remove))
                        
                        if original_text != no_punct_text:
                            item[key] = no_punct_text
                            modified = True  

                refined_data.append(item)
                if modified:
                    numbers += 1
        dataset.dataset = refined_data
        return dataset, numbers
