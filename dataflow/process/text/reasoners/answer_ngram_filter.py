from dataflow.core import ReasonerFilter
import numpy as np
import re
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.Eval.Text import NgramScorer
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class AnswerNgramFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerNgramFilter'
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        self.ngrams = args_dict['ngrams']
        self.logger = get_logger()
        
    def filter_func(self, dataset):
        scores = []
        for sample in dataset:
            answer = sample[self.question_key]
            answer += sample[self.answer_key]
            content = answer.lower()
            content = re.sub(r'[^\w\s]', '', content)
            words = content.split()
            ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
            unique_ngrams = set(ngrams)

            total_ngrams = len(ngrams)
            unique_ngrams_count = len(unique_ngrams)

            repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
            scores.append(repetition_score) 

        return np.array([self.min_score <= score <= self.max_score for score in scores]).astype(int)