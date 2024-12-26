from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from nltk.corpus import stopwords
from dataflow.utils.registry import PROCESSOR_REGISTRY
import nltk
from tqdm import tqdm

"""
The RemoveStopwordsRefiner class is a text refiner that removes common stopwords from specified text fields in a dataset.
Using NLTK’s predefined list of English stopwords, it filters out words like "the", "and", "is", etc., which are often 
considered non-essential for certain types of text analysis. 

During processing, each specified field in the dataset is checked, and any stopwords are removed, leaving only 
the more meaningful words. After cleaning, the modified dataset is returned along with a count of the items 
that were changed, resulting in a more concise and content-focused text format.
"""


@PROCESSOR_REGISTRY.register()
class RemoveStopwordsRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'RemoveStopwordsRefiner'
        nltk.data.path.append(args_dict.get('model_cache_dir'))
        nltk.download('stopwords', download_dir=args_dict.get('model_cache_dir'))
        

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
                        refined_text = self.remove_stopwords(original_text)

                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1
        dataset.dataset = refined_data
        return dataset, numbers

    def remove_stopwords(self, text):
        words = text.split()
        stopwords_list = set(stopwords.words('english'))
        refined_words = [word for word in words if word.lower() not in stopwords_list]
        return " ".join(refined_words)
