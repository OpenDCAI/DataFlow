import re
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class NgramSampleEvaluator(OperatorABC):
    
    def __init__(self, ngrams=5):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.ngrams = ngrams
        self.score_name = 'NgramScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "计算文本中n-gram的重复比例，评估文本冗余度。通过比较唯一n-gram数量与总n-gram数量的比值来衡量文本原创性。\n\n"
                "初始化参数：\n"
                "- ngrams: n-gram的长度，默认为5\n\n"
                "输出参数：\n"
                "- NgramScore: n-gram重复比例得分（0到1之间，得分越高表示重复比例越低）"
            )
        else:
            return (
                "Evaluates text redundancy by calculating n-gram repetition ratio. Measures text originality by comparing the ratio of unique n-grams to total n-grams.\n\n"
                "Initialization Parameters:\n"
                "- ngrams: Length of n-grams, default is 5\n\n"
                "Output Parameters:\n"
                "- NgramScore: N-gram repetition ratio score (0-1, higher = less repetition)"
            )
    
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
        