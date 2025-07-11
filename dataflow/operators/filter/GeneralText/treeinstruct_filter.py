import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import TreeinstructScorer

@OPERATOR_REGISTRY.register()
class TreeinstructFilter(OperatorABC):

    def __init__(self, min_score: int = 7, max_score: int = 100, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = TreeinstructScorer(llm_serving=llm_serving)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {min_score} and max_score = {max_score}")
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'TreeinstructScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
