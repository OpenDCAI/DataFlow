import re
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class NgramScorer(OperatorABC):
    
    def __init__(self, ngrams=5):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.ngrams = ngrams
        self.score_name = 'NgramScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "计算文本中n-gram的重复比例，得分越高表示重复比例越低。" if lang == "zh" else "Evaluate text redundancy via n-gram repetition; higher score means lower repetition."
    
    def _score_func(self, sample):
        content = sample 
        content = content.lower()
        content = re.sub(r'[^\w\s]', '', content)
        words = content.split()
        ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
        unique_ngrams = set(ngrams)

        total_ngrams = len(ngrams)
        unique_ngrams_count = len(unique_ngrams)

        repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
        return repetition_score

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = [self._score_func(sample) for sample in tqdm(dataframe[input_key], desc="NgramScorer Evaluating...")]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='NgramScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
        