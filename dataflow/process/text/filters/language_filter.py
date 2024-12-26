import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from dataflow.core import TextFilter
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm

@PROCESSOR_REGISTRY.register()
class LanguageFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.allowed_languages = args_dict['allowed_languages']
        model_cache_dir = args_dict.get('model_cache_dir', None)
        
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin", cache_dir=model_cache_dir)
        self.model = fasttext.load_model(model_path)
        self.filter_name = 'LanguageFilter'

    def filter_func(self, dataset):
        predictions = []
        for item in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            if isinstance(dataset.keys, list):
                text_to_evaluate = " ".join(item[key].replace('\n', ' ') for key in dataset.keys)
            else:
                text_to_evaluate = item[dataset.keys].replace('\n', ' ')
            labels, _ = self.model.predict(text_to_evaluate, k=5)
            predictions.append(any(label in self.allowed_languages for label in labels))

        return np.array(predictions).astype(int)
