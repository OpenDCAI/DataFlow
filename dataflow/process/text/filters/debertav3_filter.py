from dataflow.Eval.Text import DebertaV3Scorer
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class DebertaV3Filter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.allowed_scores = args_dict['allowed_scores']
        scorer_args = args_dict.get('scorer_args', {})
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = DebertaV3Scorer(scorer_args)
        self.filter_name = 'DebertaV3Filter'

    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        metric_scores = scores['Default']
        metric_filter = np.array([1 if score in self.allowed_scores else 0 for score in metric_scores])
        return metric_filter
