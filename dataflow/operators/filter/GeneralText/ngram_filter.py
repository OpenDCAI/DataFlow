from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import NgramScorer

@OPERATOR_REGISTRY.register()
class NgramFilter(OperatorABC):

    def __init__(self, min_score=0.8, max_score=1, ngrams=5):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = NgramScorer(ngrams)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores: {self.min_score} and max_scores: {self.max_score}...")  

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='NgramScore'):
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        