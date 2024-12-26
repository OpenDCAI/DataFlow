from dataflow.Eval.video import FastVQAScorer
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.core import VideoFilter
import numpy as np

@PROCESSOR_REGISTRY.register()
class FastVQAFilter(VideoFilter):
    
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        self.scorer = FastVQAScorer(args_dict['scorer_args'])
    
    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        return np.array([self.min_score <= score <= self.max_score for score in scores['Default'] ]).astype(int)