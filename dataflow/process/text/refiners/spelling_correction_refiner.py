from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from symspellpy.symspellpy import SymSpell, Verbosity
from dataflow.utils.registry import PROCESSOR_REGISTRY
import os
import requests
from tqdm import tqdm

"""
The SpellingCorrectionRefiner class is a text refiner that corrects spelling errors in specified text fields within a dataset.
It utilizes the SymSpell library, which provides fast spelling correction based on a frequency dictionary of English words.
If the frequency dictionary file is not found locally, the class downloads it automatically from a predefined URL.

This refiner goes through each word in the text and corrects it if a close match is found in the dictionary within a specified edit distance.
The `max_edit_distance` and `prefix_length` parameters allow configuration of the correction sensitivity. After processing,
the refined dataset is returned along with a count of modified items, providing cleaner, error-free text for analysis.

This class is particularly useful for datasets where misspellings may interfere with text analysis accuracy.
"""

@PROCESSOR_REGISTRY.register()
class SpellingCorrectionRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'SpellingCorrectionRefiner'
        self.max_edit_distance = args_dict.get('max_edit_distance', 2)  # Default to 2 if not specified
        self.prefix_length = args_dict.get('prefix_length', 7)  # Default to 7 if not specified

        self.dictionary_path = args_dict.get('dictionary_path', 'frequency_dictionary_en_82_765.txt')

        # If dictionary is not found locally, download it
        if not os.path.exists(self.dictionary_path):
            self.download_dictionary()

        self.sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=self.prefix_length)
        term_index = 0
        count_index = 1
        if not self.sym_spell.load_dictionary(self.dictionary_path, term_index, count_index):
            print(f"Error loading dictionary at {self.dictionary_path}")
    
        
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
                        refined_text = self.spelling_checks(original_text)
                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1

        dataset.dataset = refined_data
        return dataset, numbers

    def spelling_checks(self, text):
        correct_result = []
        for word in text.split():
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, self.max_edit_distance)
            corrected_word = suggestions[0].term if suggestions else word
            correct_result.append(corrected_word)

        return " ".join(correct_result)

    def download_dictionary(self):
        url = 'https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt'
        
        try:
            print("Downloading dictionary...")
            response = requests.get(url)
            response.raise_for_status() 
            
            with open(self.dictionary_path, 'wb') as file:
                file.write(response.content)
            print(f"Dictionary downloaded and saved to {self.dictionary_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dictionary: {e}")
    