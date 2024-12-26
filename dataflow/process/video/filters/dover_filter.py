from dataflow.Eval.video import DOVERScorer
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.core import VideoFilter
import numpy as np

@PROCESSOR_REGISTRY.register()
class DOVERFilter(VideoFilter):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.min_tech_score = args_dict['min_tech_score']
        self.max_tech_score = args_dict['max_tech_score']
        self.min_aes_score = args_dict['min_aes_score']
        self.max_aes_score = args_dict['max_aes_score']
        self.scorer = DOVERScorer(args_dict['scorer_args'])
    
    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        return np.array([
            self.min_tech_score <= tech_score <= self.max_tech_score and
            self.min_aes_score <= aes_score <= self.max_aes_score
                for tech_score, aes_score in zip(scores['technical'], scores['aesthetic'])
        ]).astype(int)