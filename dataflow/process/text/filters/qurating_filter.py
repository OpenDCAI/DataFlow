from dataflow.Eval.Text import QuratingScorer
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.utils import get_logger
from tqdm import tqdm


@PROCESSOR_REGISTRY.register()
class QuratingFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.min_scores = args_dict['min_scores']
        self.max_scores = args_dict['max_scores']
        scorer_args = args_dict.get('scorer_args', {})
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = QuratingScorer(scorer_args)
        self.filter_name = 'QuratingFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_scores={self.min_scores} and max_scores={self.max_scores}...")

    def filter_func(self, dataset):
        self.logger.info(f"Start running {self.filter_name}...")
        
        _, scores = self.scorer(dataset)

        # Initialize results to all valid (1)
        results = np.ones(len(dataset), dtype=int)

        # Iterate over each label to apply the filter
        for label in tqdm(self.min_scores.keys(), desc=f"Applying {self.filter_name}"):
            min_score = self.min_scores[label]
            max_score = self.max_scores[label]
            score_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
            metric_scores = np.array(scores[score_key])
            
            # Apply score filter for the current label
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)

        self.logger.info(f"Filtering completed. Total records passing filter: {sum(results)}.")
        return results