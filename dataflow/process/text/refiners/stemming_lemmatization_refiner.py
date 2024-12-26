from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

"""
The StemmingLemmatizationRefiner class is a text refiner that performs stemming or lemmatization on specified text fields 
within a dataset based on a configurable choice. The choice is controlled by a `method` parameter in the configuration, 
allowing users to select "stemming" or "lemmatization" as needed.
"""

@PROCESSOR_REGISTRY.register()
class StemmingLemmatizationRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'StemmingLemmatizationRefiner'
        self.method = args_dict.get("method", "stemming").lower()
        if self.method not in ["stemming", "lemmatization"]:
            raise ValueError("Invalid method. Choose 'stemming' or 'lemmatization'.")
        
        nltk.download('wordnet') 
        nltk.download('omw-1.4')  

    def refine_func(self, dataset):
        refined_data = []
        modified_count = 0
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        keys = dataset.keys if isinstance(dataset.keys, list) else [dataset.keys]

        for item in tqdm(dataset, desc=f"Implementing {self.refiner_name}"):
            if isinstance(item, dict):
                modified = False
                for key in keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        
                        if self.method == "stemming":
                            refined_text = " ".join([stemmer.stem(word) for word in original_text.split()])
                        elif self.method == "lemmatization":
                            refined_text = " ".join([lemmatizer.lemmatize(word) for word in original_text.split()])

                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    modified_count += 1

        dataset.dataset = refined_data
        return dataset, modified_count
