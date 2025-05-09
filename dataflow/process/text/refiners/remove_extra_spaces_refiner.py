from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemoveExtraSpacesRefiner class is a text refiner that removes extra spaces from specified text fields in a dataset.
It reduces multiple consecutive spaces to a single space and removes leading or trailing spaces, helping to normalize 
text for further processing. This is achieved by splitting and rejoining the text, ensuring consistent spacing throughout.

After cleaning, the modified dataset is returned along with a count of the modified items, resulting in a cleaner and 
more uniform text format.
"""


@PROCESSOR_REGISTRY.register()
class RemoveExtraSpacesRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveExtraSpacesRefiner'

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
                        no_extra_spaces_text = " ".join(original_text.split())
                        if original_text != no_extra_spaces_text:
                            item[key] = no_extra_spaces_text
                            modified = True  

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers
