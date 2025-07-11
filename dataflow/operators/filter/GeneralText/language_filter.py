import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from tqdm import tqdm
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class LanguageFilter(OperatorABC):

    def __init__(self, allowed_languages: list, model_cache_dir: str = None):
        self.logger = get_logger()
        self.filter_name = 'LanguageFilter'
        self.logger.info(f"Initializing {self.__class__.__name__} with allowed_languages = {allowed_languages} and model_cache_dir = {model_cache_dir}...")
        
        self.allowed_languages = allowed_languages
        self.model_cache_dir = model_cache_dir
        
        # Download and load the FastText language model
        try:
            self.logger.info("Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin", cache_dir=self.model_cache_dir)
            self.model = fasttext.load_model(model_path)
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error downloading or loading model: {e}")
            raise

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用FastText语言识别模型过滤数据" if lang == "zh" else "Filter data using FastText language identification model."

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")

        predictions = []

        # Assuming the dataframe contains the text in `input_key`
        for text in tqdm(dataframe[input_key], desc=f"Implementing {self.filter_name}"):
            labels, scores = self.model.predict(text.replace('\n', ' '), k=5)
            label_score_pairs = list(zip(labels, scores))
            label_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            top_labels = [label for label, score in label_score_pairs]
            predictions.append(any(label in self.allowed_languages for label in top_labels))

        return np.array(predictions).astype(int)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='language_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name} with input_key = {self.input_key} and output_key = {self.output_key}...")
        predictions = self.eval(dataframe, self.input_key)
        dataframe[self.output_key] = predictions
        filtered_dataframe = dataframe[dataframe[self.output_key] == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
