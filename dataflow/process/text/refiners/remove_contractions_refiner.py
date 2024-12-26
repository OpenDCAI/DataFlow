from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import contractions
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The RemoveContractionsRefiner class is a text refiner that expands contractions in specified text fields within a dataset.
Using the `contractions` library, it identifies and replaces contracted words (e.g., "can't" becomes "cannot") with their full forms.
This process helps to normalize text for consistency or further processing. After expansion, the modified dataset is returned along 
with a count of items that were modified, enabling more standardized text content.
"""

@PROCESSOR_REGISTRY.register()
class RemoveContractionsRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveContractionsRefiner'

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
                        expanded_text = contractions.fix(original_text)
                        if original_text != expanded_text:
                            item[key] = expanded_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers
