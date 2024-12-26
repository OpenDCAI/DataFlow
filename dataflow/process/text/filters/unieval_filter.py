from dataflow.Eval.Text import UnievalScorer
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class UnievalFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        
        self.min_scores = args_dict['min_scores']
        self.max_scores = args_dict['max_scores']
        scorer_args = args_dict.get('scorer_args', {})
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.metrics_to_keep = scorer_args['metrics_to_keep']
        
        if not set(self.min_scores.keys()).issubset(set(self.metrics_to_keep.keys())):
            raise ValueError("The filtering metrics must be a subset of metrics_to_keep.")
        
        if not set(self.max_scores.keys()).issubset(set(self.metrics_to_keep.keys())):
            raise ValueError("The filtering metrics must be a subset of metrics_to_keep.")
        
        self.metric_name_map = {
            'fluency': 'UniEvalFluencyScore',
            'naturalness': 'UniEvalNaturalnessScore',
            'understandability': 'UniEvalUnderstandabilityScore'
        }
        
        self.scorer = UnievalScorer(scorer_args)
        self.filter_name = 'UnievalFilter'

    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)

        results = np.ones(len(dataset), dtype=int)
        
        for metric, min_score in self.min_scores.items():
            max_score = self.max_scores[metric]
            score_key = self.metric_name_map[metric]
            
            metric_scores = np.array(scores[score_key])
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            
            results = results & metric_filter.astype(int)

        return results
