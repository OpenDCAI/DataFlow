import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import PerspectiveScorer
from dataflow.serving import PerspectiveAPIServing

@OPERATOR_REGISTRY.register()
class PerspectiveFilter(OperatorABC):
    def __init__(self, min_score: float = 0.0, max_score: float = 0.5):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {min_score} and max_score = {max_score}")
        self.min_score = min_score
        self.max_score = max_score
        self.serving = PerspectiveAPIServing(max_workers=10)
        self.scorer = PerspectiveScorer(serving=self.serving)
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerspectiveScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        metric_filter = (scores >= self.min_score) & (scores <= self.max_score)
        nan_filter = np.isnan(scores)
        metric_filter = metric_filter | nan_filter    
        filtered_dataframe = dataframe[metric_filter]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
