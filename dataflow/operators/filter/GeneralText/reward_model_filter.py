import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import RMScorer

@OPERATOR_REGISTRY.register()
class RMFilter(OperatorABC):

    def __init__(self, min_score: float = 0.2, max_score: float = 0.8, device='cuda', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = RMScorer(device=device, model_cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score}, max_score = {self.max_score}")
        
    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'RMScore'):
        self.input_instruction_key = input_instruction_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_instruction_key = {self.input_instruction_key}, intput_output_key = {self.input_output_key}, output_key = {self.output_key}...")
        scores = np.array(self.scorer.eval(dataframe, self.input_instruction_key, self.input_output_key))
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]