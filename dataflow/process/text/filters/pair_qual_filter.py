from dataflow.Eval.Text import PairQualScorer
import numpy as np
from dataflow.core import TextFilter
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class PairQualFilter(TextFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        scorer_args = args_dict.get('scorer_args')
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = PairQualScorer(scorer_args)
        self.filter_name = 'PairQualFilter'
    
    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        return np.array([self.min_score <= score <= self.max_score for score in scores['Default']]).astype(int)
