from dataflow.Eval.Text import SuperfilteringScorer
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class SuperfilteringFilter(TextFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        scorer_args = args_dict.get('scorer_args', {})
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = SuperfilteringScorer(scorer_args)
        self.filter_name = 'SuperfilteringFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_score={self.min_score} and max_score={self.max_score}...")

    def filter_func(self, dataset):
        self.logger.info(f"Start running {self.filter_name}...")
        
        # Get the scores using the scorer
        _, scores = self.scorer(dataset)

        # Apply the score filter for each record
        results = np.array([self.min_score <= score <= self.max_score for score in scores['Default']]).astype(int)

        # Log the result count
        self.logger.info(f"Filtering completed. Total records passing filter: {sum(results)}.")
        
        return results
