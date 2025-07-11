import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import PresidioScorer

@OPERATOR_REGISTRY.register()
class PresidioFilter(OperatorABC):

    def __init__(self, min_score: int = 0, max_score: int = 5, lang='en', device='cuda', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = PresidioScorer(lang=lang, device=device, model_cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}")
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PresidioScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]