import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from dataflow.core import TextFilter
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class LanguageFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.filter_name = 'LanguageFilter'
        self.logger.info(f"Initializing {self.filter_name}...")
        
        self.allowed_languages = args_dict['allowed_languages']
        model_cache_dir = args_dict.get('model_cache_dir', None)
        
        try:
            self.logger.info("Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin", cache_dir=model_cache_dir)
            self.model = fasttext.load_model(model_path)
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error downloading or loading model: {e}")
            raise
        

    def filter_func(self, dataset):
        self.logger.info(f"Start running {self.filter_name}...")
        predictions = []
        
        # Assuming dataset is a dictionary-like object with a 'keys' attribute
        for item in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            try:
                if isinstance(dataset.keys, list):
                    text_to_evaluate = " ".join(item[key].replace('\n', ' ') for key in dataset.keys)
                else:
                    text_to_evaluate = item[dataset.keys].replace('\n', ' ')

                # Predicting the language of the text
                labels, _ = self.model.predict(text_to_evaluate, k=5)
                predictions.append(any(label in self.allowed_languages for label in labels))
            except Exception as e:
                self.logger.error(f"Error processing item: {e}")
                predictions.append(0)  # Add a default value if an error occurs
        
        self.logger.info(f"Finished processing. Saving results...")
        return np.array(predictions).astype(int)
