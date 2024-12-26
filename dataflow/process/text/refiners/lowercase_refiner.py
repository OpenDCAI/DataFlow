from dataflow.core import TextRefiner
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm
"""
The LowercaseRefiner is a text refiner class that processes a dataset to convert text in specified fields to lowercase. 
It iterates through the dataset, checking each specified field in each item. If any text is found in uppercase, 
it is converted to lowercase, and the modified dataset is returned along with a count of the modified items.
"""

@PROCESSOR_REGISTRY.register()
class LowercaseRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'LowercaseRefiner'

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
                        lower_text = original_text.lower()
                        if original_text != lower_text:
                            item[key] = lower_text
                            modified = True  

                refined_data.append(item)
                if modified:
                    numbers += 1
        dataset.dataset = refined_data
        return dataset, numbers


