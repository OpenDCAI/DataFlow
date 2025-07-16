import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataflow.core import OperatorABC
from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import numpy as np

@OPERATOR_REGISTRY.register()
class FineWebEduScorer(OperatorABC):
    def __init__(self, model_cache_dir: str = './dataflow_cache', device: str = 'cuda'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = 'HuggingFaceTB/fineweb-edu-classifier'
        self.model_cache_dir = model_cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()
        self.score_name = 'FineWebEduScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "用于评估文本教育价值的Fineweb-Edu分类器，高分表示文本具有较高的教育价值。" if lang == "zh" else "Fineweb-Edu classifier for educational value; higher scores indicate more educational content."

    def _score_func(self, sample):
        tokenized_inputs = self.tokenizer(sample, return_tensors="pt", padding="longest", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy() 
        
        return logits.tolist()[0]  # Return as list for individual sample

    def eval(self, dataframe, input_key):
        scores = []
        self.logger.info(f"Evaluating {self.score_name}...")
        for sample in tqdm(dataframe[input_key], desc="Fineweb-edu model evaluating..."):
            score = self._score_func(sample)
            scores.append(score)
        self.logger.info("Evaluation complete!")
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='FinewebEduScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
