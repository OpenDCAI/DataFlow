from dataflow.Eval.Text import QuratingScorer
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class QuratingFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.min_scores = args_dict['min_scores']
        self.max_scores = args_dict['max_scores']
        scorer_args = args_dict.get('scorer_args', {})
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = QuratingScorer(scorer_args)
        self.filter_name = 'QuratingFilter'

    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)

        results = np.ones(len(dataset), dtype=int)

        for label in self.min_scores.keys():
            min_score = self.min_scores[label]
            max_score = self.max_scores[label]
            score_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
            metric_scores = np.array(scores[score_key])
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)

        return results
