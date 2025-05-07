from dataflow.Eval.Text import PairQualScorer
import numpy as np
from dataflow.core import TextFilter
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class PairQualFilter(TextFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        
        scorer_args = args_dict.get('scorer_args')
        scorer_args['model_cache_dir'] = args_dict.get('model_cache_dir')
        self.scorer = PairQualScorer(scorer_args)
        self.filter_name = 'PairQualFilter'
        
        # Log the initialization
        self.logger.info(f"Initializing {self.filter_name} with min_score={self.min_score}, max_score={self.max_score}...")

    def filter_func(self, dataset):
        self.logger.info(f"Start running {self.filter_name}...")
        
        # Run the scorer to get the scores
        _, scores = self.scorer(dataset)

        # Calculate which records pass the score filter
        valid_checks = [self.min_score <= score <= self.max_score for score in scores['Default']]

        self.logger.info(f"Filtering completed. Total records passing filter: {sum(valid_checks)}.")
        return np.array(valid_checks).astype(int)
