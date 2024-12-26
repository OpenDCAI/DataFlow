from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemoveNumberRefiner class is a text refiner designed to remove all numeric characters from specified text fields in a dataset.
It iterates through each item in the dataset, identifying fields that contain numbers, and removes these characters to leave only
non-numeric text. This is particularly useful for cases where numerical values might interfere with analysis or where text needs to be
purely alphabetical.

After processing, the refiner returns the modified dataset along with a count of the modified items, resulting in a text format free from numbers.
"""

@PROCESSOR_REGISTRY.register()
class RemoveNumberRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveNumberRefiner'

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
                        no_number_text = ''.join([char for char in original_text if not char.isdigit()])
                        if original_text != no_number_text:
                            item[key] = no_number_text
                            modified = True  

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers