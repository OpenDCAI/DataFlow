from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import re
from datetime import datetime
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The TextNormalizationRefiner class is a text refiner that standardizes date formats and currency representations within 
specified text fields in a dataset. This class performs the following transformations:

1. Converts various date formats (e.g., "MM/DD/YYYY" or "Month DD, YYYY") to a consistent "YYYY-MM-DD" format.
2. Replaces dollar amounts (e.g., "$100") with a standardized format that includes the "USD" currency symbol after the number.

These transformations ensure that date and currency information is in a uniform format, making it easier for downstream processes 
to interpret and analyze the data accurately. After normalization, the class returns the modified dataset along with a count of 
items that were changed, providing consistent and easily readable text fields.
"""

@PROCESSOR_REGISTRY.register()
class TextNormalizationRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'TextNormalizationRefiner'

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

                        refined_text = re.sub(r'(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})', r'\3-\2-\1', refined_text)
                        date_patterns = [
                            (r'\b(\w+)\s+(\d{1,2}),\s+(\d{4})\b', '%B %d, %Y'),
                            (r'\b(\d{1,2})\s+(\w+)\s+(\d{4})\b', '%d %B %Y')
                        ]
                        for pattern, date_format in date_patterns:
                            match = re.search(pattern, refined_text)
                            if match:
                                date_str = match.group(0)
                                try:
                                    parsed_date = datetime.strptime(date_str, date_format)
                                    refined_text = refined_text.replace(date_str, parsed_date.strftime('%Y-%m-%d'))
                                except ValueError:
                                    pass

                        refined_text = re.sub(r'\$\s?(\d+)', r'\1 USD', refined_text)

                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers
